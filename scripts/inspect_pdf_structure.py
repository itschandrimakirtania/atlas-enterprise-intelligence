import sys
from collections import Counter
from pathlib import Path

import pymupdf

from app.services.chunking import (
    classify_block_role,
    detect_group_rows,
    extract_document_pages,
    get_dominant_font_size,
    get_normalized_relationship_features,
    get_position_features,
    get_relative_font_size,
    get_text_shape_features,
    group_blocks_by_layout,
    order_visual_groups,
)


def find_uploaded_pdf(
    requested_filename: str,
) -> Path:
    matching_files = list(
        Path("uploads").glob(
            f"*_{requested_filename}"
        )
    )

    if not matching_files:
        raise FileNotFoundError(
            f"No uploaded PDF found matching: "
            f"{requested_filename}"
        )

    return matching_files[0]


def inspect_page(
    page,
    dominant_font_size: float,
) -> None:
    print(
        f"\n--- PAGE {page.page_number} ---"
    )

    print(
        f"Blocks: {len(page.blocks)}"
    )

    print(
        f"Page size: "
        f"{page.width:.1f} x "
        f"{page.height:.1f}"
    )

    ordered_blocks = sorted(
        page.blocks,
        key=lambda block: (
            block.bbox[1],
            block.bbox[0],
        ),
    )

    print("\nBlock roles:")

    for index, block in enumerate(
        ordered_blocks
    ):
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

        features = get_text_shape_features(
            block.text
        )

        position = get_position_features(
            block,
            page,
        )

        relationship = (
            get_normalized_relationship_features(
                block,
                following,
                page,
            )
        )

        role_result = classify_block_role(
            block=block,
            page=page,
            dominant_font_size=dominant_font_size,
            previous=previous,
            following=following,
        )

        print(
            f"  {index + 1:>2}. "
            f"text={block.text[:100]!r} | "
            f"font={block.font_size:.1f} | "
            f"relative_font="
            f"{get_relative_font_size(block, dominant_font_size):.2f} | "
            f"words="
            f"{int(features['word_count'])} | "
            f"short="
            f"{bool(features['is_short'])} | "
            f"top="
            f"{position['relative_top']:.2f} | "
            f"left="
            f"{position['relative_left']:.2f} | "
            f"bold="
            f"{block.is_bold} | "
            f"next_gap="
            f"{relationship['relative_vertical_gap']:.3f} | "
            f"role="
            f"{role_result['role']} | "
            f"score="
            f"{role_result['score']} | "
            f"reasons="
            f"{role_result['reasons']}"
        )

    visual_groups = group_blocks_by_layout(
        page
    )

    print(
        f"\nVisual groups: "
        f"{len(visual_groups)}"
    )

    for group_index, group in enumerate(
        visual_groups,
        start=1,
    ):
        print(
            f"\n  Group {group_index}:"
        )

        print(
            f"    center_x="
            f"{group.center_x / page.width:.2f}"
        )

        print(
            f"    center_y="
            f"{group.center_y / page.height:.2f}"
        )

        print(
            f"    top_y="
            f"{group.top_y / page.height:.2f}"
        )

        print(
            f"    bbox="
            f"{tuple(round(value, 1) for value in group.bbox)}"
        )

        for block in group.blocks:
            print(
                f"      - {block.text!r}"
            )

    rows = detect_group_rows(
        page,
        visual_groups,
    )

    print(
        f"\nDetected rows: "
        f"{len(rows)}"
    )

    for row_index, row in enumerate(
        rows,
        start=1,
    ):
        row_contents = [
            group.blocks[0].text
            for group in row
        ]

        print(
            f"  Row {row_index}: "
            f"{' | '.join(row_contents)}"
        )

    ordered_groups = order_visual_groups(
        page,
        visual_groups,
    )

    print("\nFinal visual-group order:")

    for index, group in enumerate(
        ordered_groups,
        start=1,
    ):
        group_text = " | ".join(
            block.text
            for block in group.blocks
        )

        print(
            f"  {index}. "
            f"{group_text!r}"
        )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: "
            "python scripts/inspect_pdf_structure.py "
            "<filename.pdf>"
        )

    requested_filename = sys.argv[1]

    pdf_path = find_uploaded_pdf(
        requested_filename
    )

    pdf = pymupdf.open(
        pdf_path
    )

    try:
        pages = extract_document_pages(
            pdf
        )
    finally:
        pdf.close()

    dominant_font_size = (
        get_dominant_font_size(pages)
    )

    if dominant_font_size is None:
        dominant_font_size = 1.0

    print(
        f"PDF: {pdf_path}"
    )

    print(
        f"Total pages: "
        f"{len(pages)}"
    )

    print(
        f"Dominant font size: "
        f"{dominant_font_size} pt"
    )

    for page in pages:
        inspect_page(
            page,
            dominant_font_size,
        )

    font_sizes = []

    for page in pages:
        for block in page.blocks:
            font_sizes.append(
                round(
                    block.font_size,
                    1,
                )
            )

    print(
        "\n--- FONT SIZE DISTRIBUTION ---"
    )

    for size, count in Counter(
        font_sizes
    ).most_common():
        print(
            f"{size:>6} pt → "
            f"{count} blocks"
        )


if __name__ == "__main__":
    main()