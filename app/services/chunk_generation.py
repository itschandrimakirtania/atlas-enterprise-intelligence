from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.services.chunking import (
    DocumentBlock,
    DocumentChunk,
    DocumentPage,
    VisualGroup,
)


class Tokenizer(Protocol):
    """
    Minimal tokenizer interface required by the chunk generator.

    The chunking layer only needs to convert text into tokens and
    reconstruct text from tokens. The actual implementation can
    later be replaced by the tokenizer belonging to the selected
    embedding model.
    """

    def encode(self, text: str) -> list[str]:
        ...

    def decode(self, tokens: Sequence[str]) -> str:
        ...


class WhitespaceTokenizer:
    """
    Deterministic tokenizer used during development and testing.

    This is intentionally not presented as the final embedding-model
    tokenizer. It gives us a stable implementation while the embedding
    model is still undecided.
    """

    def encode(self, text: str) -> list[str]:
        return text.split()

    def decode(self, tokens: Sequence[str]) -> str:
        return " ".join(tokens)


@dataclass(frozen=True)
class ChunkingConfig:
    target_tokens: int = 400
    max_tokens: int = 500
    overlap_tokens: int = 50

    def __post_init__(self) -> None:
        if self.target_tokens <= 0:
            raise ValueError(
                "target_tokens must be greater than zero."
            )

        if self.max_tokens <= 0:
            raise ValueError(
                "max_tokens must be greater than zero."
            )

        if self.target_tokens > self.max_tokens:
            raise ValueError(
                "target_tokens cannot exceed max_tokens."
            )

        if self.overlap_tokens < 0:
            raise ValueError(
                "overlap_tokens cannot be negative."
            )

        if self.overlap_tokens >= self.max_tokens:
            raise ValueError(
                "overlap_tokens must be smaller than max_tokens."
            )


@dataclass(frozen=True)
class _PreparedGroup:
    text: str
    page_start: int
    page_end: int
    section: str | None
    subsection: str | None


def _clean_text(text: str) -> str:
    return " ".join(text.split())


def _group_text(group: VisualGroup) -> str:
    return _clean_text(
        " ".join(
            block.text
            for block in group.blocks
        )
    )


def _first_block(
    group: VisualGroup,
) -> DocumentBlock | None:
    if not group.blocks:
        return None

    return group.blocks[0]


def _get_local_role_baseline(
    page: DocumentPage,
) -> float:
    """
    Estimate the page's body-text font size.

    The structural classifier needs a baseline to distinguish
    body text from headings and titles.

    When font-size frequencies tie, the smaller font size is
    preferred because body text is normally smaller than structural
    headings and titles.

    This is particularly important for small synthetic/test pages
    where every font size may occur only once.
    """

    font_sizes = [
        round(block.font_size, 1)
        for block in page.blocks
        if block.font_size > 0
    ]

    if not font_sizes:
        return 1.0

    counts = Counter(
        font_sizes
    )

    return min(
        counts,
        key=lambda size: (
            -counts[size],
            size,
        ),
    )


def _classify_group_role(
    group: VisualGroup,
    page: DocumentPage,
    baseline_font_size: float,
) -> str:
    """
    Determine the structural role of a visual group from its first
    block.

    Detailed structural inference remains owned by chunking.py.
    This function only supplies the appropriate page-local font
    baseline required by that classifier.
    """

    from app.services.chunking import classify_block_role

    first_block = _first_block(group)

    if first_block is None:
        return "body"

    ordered_blocks = sorted(
        page.blocks,
        key=lambda block: (
            block.bbox[1],
            block.bbox[0],
        ),
    )

    try:
        index = ordered_blocks.index(
            first_block
        )
    except ValueError:
        return "body"

    previous = (
        ordered_blocks[index - 1]
        if index > 0
        else None
    )

    following = (
        ordered_blocks[index + 1]
        if index + 1 < len(ordered_blocks)
        else None
    )

    result = classify_block_role(
        block=first_block,
        page=page,
        dominant_font_size=baseline_font_size,
        previous=previous,
        following=following,
    )

    return str(
        result["role"]
    )


def _build_prepared_groups(
    page: DocumentPage,
    groups: list[VisualGroup],
) -> list[_PreparedGroup]:
    if not groups:
        return []

    baseline_font_size = (
        _get_local_role_baseline(page)
    )

    prepared_groups: list[_PreparedGroup] = []

    current_section: str | None = None

    for group in groups:
        text = _group_text(group)

        if not text:
            continue

        role = _classify_group_role(
            group=group,
            page=page,
            baseline_font_size=baseline_font_size,
        )

        first_block = _first_block(group)

        if first_block is None:
            continue

        if role == "title":
            current_section = text
            continue

        if role == "metadata":
            continue

        subsection = (
            text
            if role == "heading"
            else None
        )

        prepared_groups.append(
            _PreparedGroup(
                text=text,
                page_start=first_block.page_number,
                page_end=max(
                    block.page_number
                    for block in group.blocks
                ),
                section=current_section,
                subsection=subsection,
            )
        )

    return prepared_groups


def _split_tokens(
    tokens: list[str],
    config: ChunkingConfig,
    tokenizer: Tokenizer,
) -> list[list[str]]:
    """
    Split an oversized token sequence into chunks.

    Every produced chunk is at most max_tokens tokens.

    The target_tokens value controls normal chunk growth.
    Oversized structural groups are split only when required.
    """

    if not tokens:
        return []

    if len(tokens) <= config.max_tokens:
        return [tokens]

    chunks: list[list[str]] = []

    start = 0

    while start < len(tokens):
        remaining = len(tokens) - start

        if remaining <= config.max_tokens:
            chunks.append(
                tokens[start:]
            )
            break

        end = min(
            start + config.target_tokens,
            len(tokens),
        )

        chunk = tokens[start:end]

        if len(chunk) > config.max_tokens:
            chunk = chunk[:config.max_tokens]

        chunks.append(chunk)

        next_start = (
            end - config.overlap_tokens
        )

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def _split_group(
    group: _PreparedGroup,
    config: ChunkingConfig,
    tokenizer: Tokenizer,
) -> list[DocumentChunk]:
    tokens = tokenizer.encode(
        group.text
    )

    if not tokens:
        return []

    token_groups = _split_tokens(
        tokens=tokens,
        config=config,
        tokenizer=tokenizer,
    )

    chunks: list[DocumentChunk] = []

    for token_group in token_groups:
        text = tokenizer.decode(
            token_group
        ).strip()

        if not text:
            continue

        chunks.append(
            DocumentChunk(
                text=text,
                chunk_index=0,
                page_start=group.page_start,
                page_end=group.page_end,
                section=group.section,
                subsection=group.subsection,
                token_count=len(token_group),
            )
        )

    return chunks


def _combine_small_groups(
    groups: list[_PreparedGroup],
    config: ChunkingConfig,
    tokenizer: Tokenizer,
) -> list[DocumentChunk]:
    """
    Combine adjacent structural groups when doing so is safe.

    The generator prefers structural boundaries, but a sequence of
    small body groups under the target size can be combined when
    they share the same section and the resulting chunk stays within
    the target size.

    A subsection boundary is preserved when a group explicitly
    defines one.
    """

    chunks: list[DocumentChunk] = []

    current_text_tokens: list[str] = []
    current_page_start: int | None = None
    current_page_end: int | None = None
    current_section: str | None = None
    current_subsection: str | None = None

    def flush() -> None:
        nonlocal current_text_tokens
        nonlocal current_page_start
        nonlocal current_page_end
        nonlocal current_section
        nonlocal current_subsection

        if not current_text_tokens:
            return

        text = tokenizer.decode(
            current_text_tokens
        ).strip()

        chunks.append(
            DocumentChunk(
                text=text,
                chunk_index=0,
                page_start=(
                    current_page_start
                    if current_page_start is not None
                    else 0
                ),
                page_end=(
                    current_page_end
                    if current_page_end is not None
                    else 0
                ),
                section=current_section,
                subsection=current_subsection,
                token_count=len(
                    current_text_tokens
                ),
            )
        )

        current_text_tokens = []
        current_page_start = None
        current_page_end = None
        current_section = None
        current_subsection = None

    for group in groups:
        group_tokens = tokenizer.encode(
            group.text
        )

        if not group_tokens:
            continue

        if len(group_tokens) > config.max_tokens:
            flush()

            chunks.extend(
                _split_group(
                    group=group,
                    config=config,
                    tokenizer=tokenizer,
                )
            )

            continue

        if not current_text_tokens:
            current_text_tokens = list(
                group_tokens
            )

            current_page_start = (
                group.page_start
            )

            current_page_end = (
                group.page_end
            )

            current_section = (
                group.section
            )

            current_subsection = (
                group.subsection
            )

            continue

        same_section = (
            current_section
            == group.section
        )

        no_new_subsection = (
            group.subsection is None
        )

        proposed_tokens = (
            current_text_tokens
            + group_tokens
        )

        can_combine = (
            same_section
            and no_new_subsection
            and len(proposed_tokens)
            <= config.target_tokens
        )

        if can_combine:
            current_text_tokens = (
                proposed_tokens
            )

            current_page_end = (
                group.page_end
            )

            continue

        flush()

        current_text_tokens = list(
            group_tokens
        )

        current_page_start = (
            group.page_start
        )

        current_page_end = (
            group.page_end
        )

        current_section = (
            group.section
        )

        current_subsection = (
            group.subsection
        )

    flush()

    return chunks


def _assign_chunk_indices(
    chunks: list[DocumentChunk],
) -> list[DocumentChunk]:
    return [
        DocumentChunk(
            text=chunk.text,
            chunk_index=index,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            section=chunk.section,
            subsection=chunk.subsection,
            token_count=chunk.token_count,
        )
        for index, chunk in enumerate(chunks)
    ]


def generate_page_chunks(
    page: DocumentPage,
    groups: list[VisualGroup],
    tokenizer: Tokenizer,
    config: ChunkingConfig | None = None,
) -> list[DocumentChunk]:
    """
    Generate retrieval-ready chunks for one page.

    Structural groups are the primary boundary. Small groups may be
    combined when they share section context and remain below the
    target size. Oversized groups are recursively token-split.
    """

    if config is None:
        config = ChunkingConfig()

    prepared_groups = _build_prepared_groups(
        page=page,
        groups=groups,
    )

    chunks = _combine_small_groups(
        groups=prepared_groups,
        config=config,
        tokenizer=tokenizer,
    )

    return _assign_chunk_indices(
        chunks
    )


def generate_document_chunks(
    pages: Sequence[DocumentPage],
    ordered_groups_by_page: Sequence[
        Sequence[VisualGroup]
    ],
    tokenizer: Tokenizer,
    config: ChunkingConfig | None = None,
) -> list[DocumentChunk]:
    """
    Generate retrieval-ready chunks for an entire document.

    Each page is processed independently so unrelated content at a
    page boundary is never merged accidentally.

    Chunk indices are assigned globally in final document order.
    """

    if config is None:
        config = ChunkingConfig()

    if len(pages) != len(
        ordered_groups_by_page
    ):
        raise ValueError(
            "pages and ordered_groups_by_page "
            "must have the same length."
        )

    all_chunks: list[DocumentChunk] = []

    for page, groups in zip(
        pages,
        ordered_groups_by_page,
    ):
        page_chunks = generate_page_chunks(
            page=page,
            groups=list(groups),
            tokenizer=tokenizer,
            config=config,
        )

        all_chunks.extend(
            page_chunks
        )

    return _assign_chunk_indices(
        all_chunks
    )