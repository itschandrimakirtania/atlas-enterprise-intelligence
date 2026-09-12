import pytest

from app.services.chunk_generation import (
    ChunkingConfig,
    WhitespaceTokenizer,
    generate_document_chunks,
    generate_page_chunks,
)
from app.services.chunking import (
    DocumentBlock,
    DocumentPage,
    VisualGroup,
)


def make_page(
    blocks: list[DocumentBlock],
    page_number: int = 1,
) -> DocumentPage:
    return DocumentPage(
        page_number=page_number,
        text="\n\n".join(
            block.text
            for block in blocks
        ),
        blocks=blocks,
        width=1000,
        height=1000,
    )


def make_group(
    *blocks: DocumentBlock,
) -> VisualGroup:
    center_x = sum(
        (
            block.bbox[0]
            + block.bbox[2]
        ) / 2
        for block in blocks
    ) / len(blocks)

    center_y = sum(
        (
            block.bbox[1]
            + block.bbox[3]
        ) / 2
        for block in blocks
    ) / len(blocks)

    top_y = min(
        block.bbox[1]
        for block in blocks
    )

    bbox = (
        min(block.bbox[0] for block in blocks),
        min(block.bbox[1] for block in blocks),
        max(block.bbox[2] for block in blocks),
        max(block.bbox[3] for block in blocks),
    )

    return VisualGroup(
        blocks=list(blocks),
        center_x=center_x,
        center_y=center_y,
        top_y=top_y,
        bbox=bbox,
    )


def test_small_heading_body_group_stays_together():
    heading = DocumentBlock(
        text="Asset Management",
        page_number=1,
        font_size=26.0,
        is_bold=True,
        bbox=(40, 200, 300, 240),
    )

    body = DocumentBlock(
        text=(
            "Asset management helps organizations "
            "track and maintain resources efficiently."
        ),
        page_number=1,
        font_size=21.0,
        is_bold=False,
        bbox=(40, 280, 900, 350),
    )

    page = make_page(
        [heading, body]
    )

    groups = [
        make_group(
            heading,
            body,
        )
    ]

    chunks = generate_page_chunks(
        page=page,
        groups=groups,
        tokenizer=WhitespaceTokenizer(),
    )

    assert len(chunks) == 1
    assert chunks[0].text.startswith(
        "Asset Management"
    )
    assert chunks[0].subsection == (
        "Asset Management Asset management helps organizations "
        "track and maintain resources efficiently."
    )
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1


def test_title_is_used_as_section_context_not_as_retrieval_chunk():
    title = DocumentBlock(
        text="Applications and Use Cases",
        page_number=1,
        font_size=80.0,
        is_bold=False,
        bbox=(40, 50, 700, 130),
    )

    heading = DocumentBlock(
        text="IT Asset Tracking",
        page_number=1,
        font_size=26.0,
        is_bold=False,
        bbox=(40, 200, 300, 240),
    )

    body = DocumentBlock(
        text=(
            "Cloud systems allow organizations to "
            "monitor asset lifecycle and status."
        ),
        page_number=1,
        font_size=21.0,
        is_bold=False,
        bbox=(40, 280, 900, 350),
    )

    page = make_page(
        [title, heading, body]
    )

    groups = [
        make_group(title),
        make_group(heading, body),
    ]

    chunks = generate_page_chunks(
        page=page,
        groups=groups,
        tokenizer=WhitespaceTokenizer(),
    )

    assert len(chunks) == 1
    assert chunks[0].section == (
        "Applications and Use Cases"
    )
    assert chunks[0].subsection == (
        "IT Asset Tracking Cloud systems allow organizations to "
        "monitor asset lifecycle and status."
    )
    assert "Applications and Use Cases" not in (
        chunks[0].text
    )


def test_metadata_group_is_excluded():
    title = DocumentBlock(
        text="Cloud Asset Management",
        page_number=1,
        font_size=80.0,
        is_bold=False,
        bbox=(40, 50, 700, 130),
    )

    metadata = DocumentBlock(
        text="Presented by CHANDRIMA KIRTANIA",
        page_number=1,
        font_size=26.0,
        is_bold=False,
        bbox=(40, 850, 500, 890),
    )

    body = DocumentBlock(
        text=(
            "Cloud asset management provides centralized "
            "visibility into enterprise resources."
        ),
        page_number=1,
        font_size=21.0,
        is_bold=False,
        bbox=(40, 200, 900, 280),
    )

    page = make_page(
        [title, body, metadata]
    )

    groups = [
        make_group(title),
        make_group(body),
        make_group(metadata),
    ]

    chunks = generate_page_chunks(
        page=page,
        groups=groups,
        tokenizer=WhitespaceTokenizer(),
    )

    assert len(chunks) == 1
    assert (
        "Presented by"
        not in chunks[0].text
    )


def test_oversized_group_is_split_below_hard_maximum():
    heading = DocumentBlock(
        text="Large Section",
        page_number=1,
        font_size=26.0,
        is_bold=True,
        bbox=(40, 100, 300, 140),
    )

    body = DocumentBlock(
        text=" ".join(
            f"token{i}"
            for i in range(650)
        ),
        page_number=1,
        font_size=21.0,
        is_bold=False,
        bbox=(40, 180, 900, 800),
    )

    page = make_page(
        [heading, body]
    )

    groups = [
        make_group(
            heading,
            body,
        )
    ]

    config = ChunkingConfig(
        target_tokens=400,
        max_tokens=500,
        overlap_tokens=50,
    )

    chunks = generate_page_chunks(
        page=page,
        groups=groups,
        tokenizer=WhitespaceTokenizer(),
        config=config,
    )

    assert len(chunks) >= 2

    assert all(
        chunk.token_count <= 500
        for chunk in chunks
    )


def test_oversized_split_preserves_overlap():
    body = DocumentBlock(
        text=" ".join(
            f"token{i}"
            for i in range(900)
        ),
        page_number=1,
        font_size=21.0,
        is_bold=False,
        bbox=(40, 100, 900, 800),
    )

    page = make_page(
        [body]
    )

    groups = [
        make_group(body)
    ]

    config = ChunkingConfig(
        target_tokens=400,
        max_tokens=500,
        overlap_tokens=50,
    )

    chunks = generate_page_chunks(
        page=page,
        groups=groups,
        tokenizer=WhitespaceTokenizer(),
        config=config,
    )

    assert len(chunks) == 3

    first_tokens = chunks[0].text.split()
    second_tokens = chunks[1].text.split()

    assert (
        first_tokens[-50:]
        == second_tokens[:50]
    )


def test_structural_groups_do_not_merge_across_subsections():
    first_heading = DocumentBlock(
        text="IT Asset Tracking",
        page_number=1,
        font_size=26.0,
        is_bold=False,
        bbox=(40, 100, 300, 140),
    )

    first_body = DocumentBlock(
        text=" ".join(
            f"tracking{i}"
            for i in range(40)
        ),
        page_number=1,
        font_size=21.0,
        is_bold=False,
        bbox=(40, 180, 900, 300),
    )

    second_heading = DocumentBlock(
        text="Facility Monitoring",
        page_number=1,
        font_size=26.0,
        is_bold=False,
        bbox=(40, 400, 300, 440),
    )

    second_body = DocumentBlock(
        text=" ".join(
            f"facility{i}"
            for i in range(40)
        ),
        page_number=1,
        font_size=21.0,
        is_bold=False,
        bbox=(40, 480, 900, 600),
    )

    page = make_page(
        [
            first_heading,
            first_body,
            second_heading,
            second_body,
        ]
    )

    groups = [
        make_group(
            first_heading,
            first_body,
        ),
        make_group(
            second_heading,
            second_body,
        ),
    ]

    chunks = generate_page_chunks(
        page=page,
        groups=groups,
        tokenizer=WhitespaceTokenizer(),
    )

    assert len(chunks) == 2

    assert chunks[0].subsection.startswith(
        "IT Asset Tracking"
    )

    assert chunks[1].subsection.startswith(
        "Facility Monitoring"
    )


def test_page_boundaries_are_preserved():
    page_one_body = DocumentBlock(
        text="Information from page one.",
        page_number=1,
        font_size=21.0,
        is_bold=False,
        bbox=(40, 200, 900, 300),
    )

    page_two_body = DocumentBlock(
        text="Information from page two.",
        page_number=2,
        font_size=21.0,
        is_bold=False,
        bbox=(40, 200, 900, 300),
    )

    page_one = make_page(
        [page_one_body],
        page_number=1,
    )

    page_two = make_page(
        [page_two_body],
        page_number=2,
    )

    groups_one = [
        make_group(page_one_body)
    ]

    groups_two = [
        make_group(page_two_body)
    ]

    chunks = generate_document_chunks(
        pages=[
            page_one,
            page_two,
        ],
        ordered_groups_by_page=[
            groups_one,
            groups_two,
        ],
        tokenizer=WhitespaceTokenizer(),
    )

    assert len(chunks) == 2

    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1

    assert chunks[1].page_start == 2
    assert chunks[1].page_end == 2


def test_global_chunk_indices_are_sequential():
    body_one = DocumentBlock(
        text="Page one information.",
        page_number=1,
        font_size=21.0,
        is_bold=False,
        bbox=(40, 200, 900, 300),
    )

    body_two = DocumentBlock(
        text="Page two information.",
        page_number=2,
        font_size=21.0,
        is_bold=False,
        bbox=(40, 200, 900, 300),
    )

    page_one = make_page(
        [body_one],
        page_number=1,
    )

    page_two = make_page(
        [body_two],
        page_number=2,
    )

    chunks = generate_document_chunks(
        pages=[
            page_one,
            page_two,
        ],
        ordered_groups_by_page=[
            [make_group(body_one)],
            [make_group(body_two)],
        ],
        tokenizer=WhitespaceTokenizer(),
    )

    assert [
        chunk.chunk_index
        for chunk in chunks
    ] == [0, 1]


def test_invalid_chunk_configuration_is_rejected():
    with pytest.raises(ValueError):
        ChunkingConfig(
            target_tokens=600,
            max_tokens=500,
        )

    with pytest.raises(ValueError):
        ChunkingConfig(
            target_tokens=400,
            max_tokens=500,
            overlap_tokens=500,
        )

    with pytest.raises(ValueError):
        ChunkingConfig(
            target_tokens=0,
            max_tokens=500,
        )