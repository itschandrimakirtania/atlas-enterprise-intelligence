from collections import Counter
from dataclasses import dataclass


BBox = tuple[float, float, float, float]


@dataclass
class DocumentBlock:
    text: str
    page_number: int
    font_size: float
    is_bold: bool
    bbox: BBox


@dataclass
class VisualGroup:
    blocks: list[DocumentBlock]
    center_x: float
    center_y: float
    top_y: float
    bbox: BBox


@dataclass
class DocumentPage:
    page_number: int
    text: str
    blocks: list[DocumentBlock]
    width: float
    height: float


@dataclass
class DocumentChunk:
    text: str
    chunk_index: int
    page_start: int
    page_end: int
    section: str | None = None
    subsection: str | None = None
    token_count: int = 0


def extract_document_pages(pdf) -> list[DocumentPage]:
    pages = []

    for page_number, page in enumerate(pdf, start=1):
        page_data = page.get_text("dict")
        blocks = []

        for block in page_data.get("blocks", []):
            if block.get("type") != 0:
                continue

            block_text_parts = []
            font_sizes = []
            bold_flags = []

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()

                    if not text:
                        continue

                    block_text_parts.append(text)
                    font_sizes.append(span.get("size", 0.0))

                    font_flags = span.get("flags", 0)
                    bold_flags.append(bool(font_flags & 16))

            if not block_text_parts:
                continue

            block_text = " ".join(block_text_parts)

            average_font_size = (
                sum(font_sizes) / len(font_sizes)
            )

            is_bold = (
                sum(bold_flags)
                > len(bold_flags) / 2
            )

            blocks.append(
                DocumentBlock(
                    text=block_text,
                    page_number=page_number,
                    font_size=average_font_size,
                    is_bold=is_bold,
                    bbox=tuple(block["bbox"]),
                )
            )

        page_text = "\n\n".join(
            block.text
            for block in blocks
        )

        pages.append(
            DocumentPage(
                page_number=page_number,
                text=page_text,
                blocks=blocks,
                width=page.rect.width,
                height=page.rect.height,
            )
        )

    return pages


def get_dominant_font_size(
    pages: list[DocumentPage],
) -> float | None:
    font_sizes = []

    for page in pages:
        for block in page.blocks:
            font_sizes.append(
                round(block.font_size, 1)
            )

    if not font_sizes:
        return None

    return Counter(
        font_sizes
    ).most_common(1)[0][0]


def get_relative_font_size(
    block: DocumentBlock,
    dominant_font_size: float,
) -> float:
    if dominant_font_size <= 0:
        return 1.0

    return block.font_size / dominant_font_size


def get_text_shape_features(
    text: str,
) -> dict[str, float]:
    cleaned_text = " ".join(
        text.split()
    )

    if not cleaned_text:
        return {
            "word_count": 0,
            "character_count": 0,
            "has_sentence_punctuation": 0.0,
            "is_short": 0.0,
        }

    words = cleaned_text.split()

    has_sentence_punctuation = float(
        cleaned_text.endswith(
            (".", "?", "!")
        )
    )

    is_short = float(
        len(words) <= 8
    )

    return {
        "word_count": float(len(words)),
        "character_count": float(
            len(cleaned_text)
        ),
        "has_sentence_punctuation": (
            has_sentence_punctuation
        ),
        "is_short": is_short,
    }


def get_position_features(
    block: DocumentBlock,
    page: DocumentPage,
) -> dict[str, float]:
    x0, y0, x1, y1 = block.bbox

    if (
        page.width <= 0
        or page.height <= 0
    ):
        return {
            "relative_left": 0.0,
            "relative_top": 0.0,
            "relative_width": 0.0,
            "relative_height": 0.0,
            "relative_center_x": 0.0,
        }

    return {
        "relative_left": (
            x0 / page.width
        ),
        "relative_top": (
            y0 / page.height
        ),
        "relative_width": (
            (x1 - x0) / page.width
        ),
        "relative_height": (
            (y1 - y0) / page.height
        ),
        "relative_center_x": (
            ((x0 + x1) / 2)
            / page.width
        ),
    }


def get_block_relationship_features(
    current: DocumentBlock,
    following: DocumentBlock | None,
) -> dict[str, float]:
    if following is None:
        return {
            "vertical_gap": 0.0,
            "horizontal_overlap": 0.0,
        }

    (
        current_x0,
        current_y0,
        current_x1,
        current_y1,
    ) = current.bbox

    (
        following_x0,
        following_y0,
        following_x1,
        following_y1,
    ) = following.bbox

    vertical_gap = max(
        0.0,
        following_y0 - current_y1,
    )

    overlap_start = max(
        current_x0,
        following_x0,
    )

    overlap_end = min(
        current_x1,
        following_x1,
    )

    horizontal_overlap = max(
        0.0,
        overlap_end - overlap_start,
    )

    current_width = max(
        0.0,
        current_x1 - current_x0,
    )

    if current_width > 0:
        horizontal_overlap /= current_width

    return {
        "vertical_gap": vertical_gap,
        "horizontal_overlap": horizontal_overlap,
    }


def get_normalized_relationship_features(
    current: DocumentBlock,
    following: DocumentBlock | None,
    page: DocumentPage,
) -> dict[str, float]:
    relationship = (
        get_block_relationship_features(
            current,
            following,
        )
    )

    if page.height <= 0:
        return {
            "relative_vertical_gap": 0.0,
            "horizontal_overlap": (
                relationship[
                    "horizontal_overlap"
                ]
            ),
        }

    return {
        "relative_vertical_gap": (
            relationship["vertical_gap"]
            / page.height
        ),
        "horizontal_overlap": (
            relationship[
                "horizontal_overlap"
            ]
        ),
    }


def classify_block_role(
    block: DocumentBlock,
    page: DocumentPage,
    dominant_font_size: float,
    previous: DocumentBlock | None,
    following: DocumentBlock | None,
) -> dict[str, object]:
    text_features = get_text_shape_features(
        block.text
    )

    position_features = get_position_features(
        block,
        page,
    )

    previous_relationship = (
        get_normalized_relationship_features(
            previous,
            block,
            page,
        )
        if previous is not None
        else {
            "relative_vertical_gap": 0.0,
            "horizontal_overlap": 0.0,
        }
    )

    following_relationship = (
        get_normalized_relationship_features(
            block,
            following,
            page,
        )
    )

    relative_font_size = (
        get_relative_font_size(
            block,
            dominant_font_size,
        )
    )

    score = 0
    reasons = []

    word_count = int(
        text_features["word_count"]
    )

    is_short = bool(
        text_features["is_short"]
    )

    has_sentence_punctuation = bool(
        text_features[
            "has_sentence_punctuation"
        ]
    )

    next_gap = following_relationship[
        "relative_vertical_gap"
    ]

    previous_gap = previous_relationship[
        "relative_vertical_gap"
    ]

    relative_top = position_features[
        "relative_top"
    ]

    # Strong typographic evidence.
    if relative_font_size >= 2.5:
        score += 5
        reasons.append(
            "very_large_font"
        )
    elif relative_font_size >= 1.10:
        score += 2
        reasons.append(
            "larger_font"
        )

    # Short blocks are structurally more likely to be
    # titles/headings than body paragraphs.
    if is_short:
        score += 2
        reasons.append(
            "short_text"
        )

    # Boldness is supporting evidence, not a decisive rule.
    if block.is_bold:
        score += 1
        reasons.append(
            "bold_text"
        )

    # Long prose should strongly resist heading classification.
    if word_count >= 15:
        score -= 3
        reasons.append(
            "long_text"
        )

    # Sentence-like prose is additional body evidence.
    if (
        has_sentence_punctuation
        and word_count >= 15
    ):
        score -= 2
        reasons.append(
            "body_like_sentence"
        )

    # Spatial separation is useful, but deliberately weak.
    # Large gaps alone should never turn prose into a heading.
    if (
        next_gap >= 0.04
        and is_short
    ):
        score += 1
        reasons.append(
            "large_following_gap"
        )

    if (
        previous_gap >= 0.04
        and is_short
    ):
        score += 1
        reasons.append(
            "large_previous_gap"
        )

    if (
        relative_top <= 0.20
        and relative_font_size >= 2.5
    ):
        score += 1
        reasons.append(
            "top_page_large_text"
        )

    # Metadata is detected from reusable spatial/typographic
    # characteristics rather than a specific phrase or page.
    is_bottom_metadata = (
        relative_top >= 0.80
        and is_short
        and relative_font_size < 2.5
    )

    if is_bottom_metadata:
        role = "metadata"
        reasons.append(
            "bottom_page_short_text"
        )
    elif relative_font_size >= 2.5:
        role = "title"
    elif (
        score >= 3
        and word_count <= 14
    ):
        role = "heading"
    else:
        role = "body"

    return {
        "role": role,
        "score": score,
        "reasons": reasons,
    }


def _vertical_overlap_ratio(
    first: DocumentBlock,
    second: DocumentBlock,
) -> float:
    first_y0 = first.bbox[1]
    first_y1 = first.bbox[3]

    second_y0 = second.bbox[1]
    second_y1 = second.bbox[3]

    overlap_start = max(
        first_y0,
        second_y0,
    )

    overlap_end = min(
        first_y1,
        second_y1,
    )

    overlap = max(
        0.0,
        overlap_end - overlap_start,
    )

    first_height = max(
        0.0,
        first_y1 - first_y0,
    )

    if first_height <= 0:
        return 0.0

    return overlap / first_height


def _horizontal_compatibility(
    first: DocumentBlock,
    second: DocumentBlock,
    page: DocumentPage,
) -> bool:
    first_x0, _, first_x1, _ = first.bbox
    second_x0, _, second_x1, _ = second.bbox

    overlap_start = max(
        first_x0,
        second_x0,
    )

    overlap_end = min(
        first_x1,
        second_x1,
    )

    overlap = max(
        0.0,
        overlap_end - overlap_start,
    )

    first_width = max(
        0.0,
        first_x1 - first_x0,
    )

    second_width = max(
        0.0,
        second_x1 - second_x0,
    )

    overlap_ratio_first = (
        overlap / first_width
        if first_width > 0
        else 0.0
    )

    overlap_ratio_second = (
        overlap / second_width
        if second_width > 0
        else 0.0
    )

    first_center = (
        first_x0 + first_x1
    ) / 2

    second_center = (
        second_x0 + second_x1
    ) / 2

    center_distance = abs(
        first_center - second_center
    )

    return (
        overlap_ratio_first >= 0.30
        or overlap_ratio_second >= 0.30
        or center_distance
        <= page.width * 0.20
    )


def _build_visual_group(
    blocks: list[DocumentBlock],
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

    group_x0 = min(
        block.bbox[0]
        for block in blocks
    )

    group_y0 = min(
        block.bbox[1]
        for block in blocks
    )

    group_x1 = max(
        block.bbox[2]
        for block in blocks
    )

    group_y1 = max(
        block.bbox[3]
        for block in blocks
    )

    return VisualGroup(
        blocks=blocks,
        center_x=center_x,
        center_y=center_y,
        top_y=top_y,
        bbox=(
            group_x0,
            group_y0,
            group_x1,
            group_y1,
        ),
    )


def group_blocks_by_layout(
    page: DocumentPage,
) -> list[VisualGroup]:
    """
    Group structurally related blocks into content units.

    Headings act as anchors. Supporting body blocks are attached
    using continuity: after a body block is attached, that block
    becomes the new geometric boundary for finding subsequent
    supporting content.

    This avoids relying on one fixed distance from the original
    heading and prevents unrelated columns from being merged.
    """
    if not page.blocks:
        return []

    dominant_font_size = (
        get_dominant_font_size([page])
    )

    if (
        dominant_font_size is None
        or dominant_font_size <= 0
    ):
        dominant_font_size = 1.0

    blocks = sorted(
        page.blocks,
        key=lambda block: (
            block.bbox[1],
            block.bbox[0],
        ),
    )

    role_results: dict[int, dict[str, object]] = {}

    for index, block in enumerate(blocks):
        previous = (
            blocks[index - 1]
            if index > 0
            else None
        )

        following = (
            blocks[index + 1]
            if index + 1 < len(blocks)
            else None
        )

        role_results[id(block)] = (
            classify_block_role(
                block=block,
                page=page,
                dominant_font_size=dominant_font_size,
                previous=previous,
                following=following,
            )
        )

    anchors = [
        block
        for block in blocks
        if role_results[id(block)]["role"]
        in {"title", "heading"}
    ]

    supporting_blocks = [
        block
        for block in blocks
        if role_results[id(block)]["role"]
        == "body"
    ]

    metadata_blocks = [
        block
        for block in blocks
        if role_results[id(block)]["role"]
        == "metadata"
    ]

    assigned: set[int] = set()
    visual_groups: list[VisualGroup] = []

    # Titles are page-level structural elements.
    # They remain separate from ordinary content groups.
    for anchor in anchors:
        role = role_results[id(anchor)]["role"]

        if role == "title":
            visual_groups.append(
                _build_visual_group(
                    [anchor]
                )
            )
            assigned.add(id(anchor))

    content_anchors = [
        anchor
        for anchor in anchors
        if id(anchor) not in assigned
    ]

    for anchor in content_anchors:
        group_blocks = [anchor]
        assigned.add(id(anchor))

        current_block = anchor

        while True:
            candidates = []

            for block in supporting_blocks:
                block_id = id(block)

                if block_id in assigned:
                    continue

                if block.bbox[1] < current_block.bbox[3]:
                    continue

                if not _horizontal_compatibility(
                    current_block,
                    block,
                    page,
                ):
                    continue

                vertical_gap = (
                    block.bbox[1]
                    - current_block.bbox[3]
                )

                relative_gap = (
                    vertical_gap
                    / page.height
                    if page.height > 0
                    else 0.0
                )

                # Body continuity can tolerate more distance than
                # heading-to-body attachment, but not arbitrary gaps.
                if relative_gap > 0.12:
                    continue

                # Prevent a later heading/anchor from being swallowed.
                competing_anchors = [
                    other
                    for other in content_anchors
                    if id(other) not in assigned
                    and other.bbox[1]
                    >= current_block.bbox[3]
                    and other.bbox[1]
                    <= block.bbox[1]
                    and _horizontal_compatibility(
                        current_block,
                        other,
                        page,
                    )
                ]

                if competing_anchors:
                    continue

                candidates.append(
                    (
                        relative_gap,
                        abs(
                            (
                                block.bbox[0]
                                + block.bbox[2]
                            ) / 2
                            - (
                                current_block.bbox[0]
                                + current_block.bbox[2]
                            ) / 2
                        ),
                        block,
                    )
                )

            if not candidates:
                break

            candidates.sort(
                key=lambda item: (
                    item[0],
                    item[1],
                )
            )

            selected = candidates[0][2]

            group_blocks.append(
                selected
            )

            assigned.add(
                id(selected)
            )

            current_block = selected

        visual_groups.append(
            _build_visual_group(
                group_blocks
            )
        )

    # Any unassigned body blocks remain standalone rather than
    # disappearing.
    for block in supporting_blocks:
        if id(block) in assigned:
            continue

        visual_groups.append(
            _build_visual_group(
                [block]
            )
        )

        assigned.add(id(block))

    # Metadata is intentionally excluded from retrieval groups.
    # This keeps presentation/author/footer information from
    # contaminating knowledge-bearing chunks.
    _ = metadata_blocks

    visual_groups.sort(
        key=lambda group: (
            group.top_y,
            group.center_x,
        )
    )

    return visual_groups


def detect_group_rows(
    page: DocumentPage,
    groups: list[VisualGroup],
) -> list[list[VisualGroup]]:
    """
    Detect approximate horizontal rows of visual groups.

    Groups belong to the same row when either their vertical
    regions overlap or their vertical centers are sufficiently
    close relative to page height.
    """
    if not groups:
        return []

    if len(groups) == 1:
        return [groups]

    row_tolerance = page.height * 0.08

    rows: list[list[VisualGroup]] = []

    groups_by_y = sorted(
        groups,
        key=lambda group: group.center_y,
    )

    for group in groups_by_y:
        assigned_row = None

        group_y0 = group.bbox[1]
        group_y1 = group.bbox[3]

        for row in rows:
            row_y0 = min(
                item.bbox[1]
                for item in row
            )

            row_y1 = max(
                item.bbox[3]
                for item in row
            )

            row_center_y = sum(
                item.center_y
                for item in row
            ) / len(row)

            vertical_overlap = (
                min(group_y1, row_y1)
                - max(group_y0, row_y0)
            )

            if (
                vertical_overlap > 0
                or abs(
                    group.center_y
                    - row_center_y
                ) <= row_tolerance
            ):
                assigned_row = row
                break

        if assigned_row is None:
            rows.append([group])
        else:
            assigned_row.append(group)

    for row in rows:
        row.sort(
            key=lambda group: group.center_x
        )

    rows.sort(
        key=lambda row: sum(
            group.center_y
            for group in row
        ) / len(row)
    )

    return rows


def _cluster_groups_by_x(
    page: DocumentPage,
    groups: list[VisualGroup],
) -> list[list[VisualGroup]]:
    if not groups:
        return []

    sorted_groups = sorted(
        groups,
        key=lambda group: group.center_x,
    )

    tolerance = page.width * 0.20

    regions: list[list[VisualGroup]] = []

    for group in sorted_groups:
        assigned_region = None

        for region in regions:
            region_center_x = sum(
                item.center_x
                for item in region
            ) / len(region)

            if (
                abs(
                    group.center_x
                    - region_center_x
                )
                <= tolerance
            ):
                assigned_region = region
                break

        if assigned_region is None:
            regions.append([group])
        else:
            assigned_region.append(group)

    return regions


def order_visual_groups(
    page: DocumentPage,
    groups: list[VisualGroup],
) -> list[VisualGroup]:
    """
    Order visual groups according to document geometry.

    Default behavior is row-major:
    top-to-bottom, then left-to-right.

    A geometry-derived mixed-layout rule handles pages where a
    dominant wide region sits beside a vertically stacked narrow
    region, producing left-region-first reading order.
    """
    if not groups:
        return []

    if len(groups) == 1:
        return groups

    rows = detect_group_rows(
        page,
        groups,
    )

    if not rows:
        return []

    row_major = [
        group
        for row in rows
        for group in row
    ]

    regions = _cluster_groups_by_x(
        page,
        groups,
    )

    if len(regions) == 2:
        regions.sort(
            key=lambda region: sum(
                item.center_x
                for item in region
            ) / len(region)
        )

        left_region = regions[0]
        right_region = regions[1]

        left_width = sum(
            max(
                0.0,
                group.bbox[2]
                - group.bbox[0],
            )
            for group in left_region
        ) / len(left_region)

        right_width = sum(
            max(
                0.0,
                group.bbox[2]
                - group.bbox[0],
            )
            for group in right_region
        ) / len(right_region)

        right_is_stacked = (
            len(right_region) >= 2
        )

        left_is_dominant = (
            left_width
            >= right_width * 1.25
        )

        if (
            right_is_stacked
            and left_is_dominant
        ):
            ordered_left = sorted(
                left_region,
                key=lambda group: (
                    group.top_y,
                    group.center_x,
                ),
            )

            ordered_right = sorted(
                right_region,
                key=lambda group: (
                    group.top_y,
                    group.center_x,
                ),
            )

            return (
                ordered_left
                + ordered_right
            )

    return row_major