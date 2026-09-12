from app.services.chunking import (
    DocumentBlock,
    DocumentPage,
    VisualGroup,
    classify_block_role,
    detect_group_rows,
    get_normalized_relationship_features,
    group_blocks_by_layout,
    order_visual_groups,
)


def make_page(
    block: DocumentBlock,
) -> DocumentPage:
    return DocumentPage(
        page_number=1,
        text=block.text,
        blocks=[block],
        width=1000,
        height=1000,
    )


def test_large_title_is_classified_as_title():
    block = DocumentBlock(
        text="Cloud Software for Asset Management",
        page_number=1,
        font_size=100.0,
        is_bold=False,
        bbox=(40, 100, 500, 250),
    )

    page = make_page(block)

    result = classify_block_role(
        block=block,
        page=page,
        dominant_font_size=21.0,
        previous=None,
        following=None,
    )

    assert result["role"] == "title"


def test_short_larger_text_is_classified_as_heading():
    block = DocumentBlock(
        text="Definition",
        page_number=1,
        font_size=26.0,
        is_bold=False,
        bbox=(40, 400, 200, 430),
    )

    following = DocumentBlock(
        text="Cloud computing is a revolutionary technology.",
        page_number=1,
        font_size=21.0,
        is_bold=False,
        bbox=(40, 480, 900, 540),
    )

    page = DocumentPage(
        page_number=1,
        text=f"{block.text}\n\n{following.text}",
        blocks=[block, following],
        width=1000,
        height=1000,
    )

    result = classify_block_role(
        block=block,
        page=page,
        dominant_font_size=21.0,
        previous=None,
        following=following,
    )

    assert result["role"] == "heading"


def test_long_body_text_is_classified_as_body():
    block = DocumentBlock(
        text=(
            "Cloud computing is a revolutionary technology that "
            "delivers computing services over the internet and "
            "allows organizations to access scalable resources "
            "without requiring organizations to maintain all "
            "underlying physical infrastructure themselves."
        ),
        page_number=1,
        font_size=21.0,
        is_bold=False,
        bbox=(40, 500, 900, 600),
    )

    page = make_page(block)

    result = classify_block_role(
        block=block,
        page=page,
        dominant_font_size=21.0,
        previous=None,
        following=None,
    )

    assert result["role"] == "body"


def test_bottom_short_text_is_classified_as_metadata():
    block = DocumentBlock(
        text="Presented by CHANDRIMA KIRTANIA",
        page_number=3,
        font_size=26.0,
        is_bold=False,
        bbox=(40, 850, 400, 890),
    )

    page = DocumentPage(
        page_number=3,
        text=block.text,
        blocks=[block],
        width=1000,
        height=1000,
    )

    result = classify_block_role(
        block=block,
        page=page,
        dominant_font_size=21.0,
        previous=None,
        following=None,
    )

    assert result["role"] == "metadata"


def test_presentation_title_is_classified_correctly():
    block = DocumentBlock(
        text="Cloud Software for Asset Management",
        page_number=1,
        font_size=100.0,
        is_bold=False,
        bbox=(57, 121, 790, 420),
    )

    following = DocumentBlock(
        text="Presented by CHANDRIMA KIRTANIA",
        page_number=1,
        font_size=26.1,
        is_bold=False,
        bbox=(57, 720, 500, 755),
    )

    page = DocumentPage(
        page_number=1,
        text=f"{block.text}\n\n{following.text}",
        blocks=[block, following],
        width=1440,
        height=810,
    )

    result = classify_block_role(
        block=block,
        page=page,
        dominant_font_size=21.0,
        previous=None,
        following=following,
    )

    assert result["role"] == "title"


def test_presentation_metadata_is_classified_correctly():
    previous = DocumentBlock(
        text="Cloud Software for Asset Management",
        page_number=1,
        font_size=100.0,
        is_bold=False,
        bbox=(57, 121, 790, 420),
    )

    block = DocumentBlock(
        text="Presented by CHANDRIMA KIRTANIA",
        page_number=1,
        font_size=26.1,
        is_bold=False,
        bbox=(57, 720, 500, 755),
    )

    page = DocumentPage(
        page_number=1,
        text=f"{previous.text}\n\n{block.text}",
        blocks=[previous, block],
        width=1440,
        height=810,
    )

    result = classify_block_role(
        block=block,
        page=page,
        dominant_font_size=21.0,
        previous=previous,
        following=None,
    )

    assert result["role"] == "metadata"


def test_presentation_section_heading_is_classified_correctly():
    block = DocumentBlock(
        text="What is Asset Management?",
        page_number=2,
        font_size=24.0,
        is_bold=False,
        bbox=(57, 405, 300, 440),
    )

    following = DocumentBlock(
        text=(
            "Asset management is the process of overseeing and "
            "managing an organization's assets efficiently."
        ),
        page_number=2,
        font_size=19.4,
        is_bold=False,
        bbox=(57, 480, 500, 550),
    )

    page = DocumentPage(
        page_number=2,
        text=f"{block.text}\n\n{following.text}",
        blocks=[block, following],
        width=1440,
        height=810,
    )

    result = classify_block_role(
        block=block,
        page=page,
        dominant_font_size=21.0,
        previous=None,
        following=following,
    )

    assert result["role"] == "heading"


def test_multi_column_heading_without_gap_is_classified_correctly():
    previous = DocumentBlock(
        text=(
            "Unlike traditional methods that relied on manual tracking, "
            "modern asset management leverages digital tools."
        ),
        page_number=2,
        font_size=19.4,
        is_bold=False,
        bbox=(965, 470, 1300, 540),
    )

    block = DocumentBlock(
        text="Traditional vs Modern",
        page_number=2,
        font_size=24.0,
        is_bold=False,
        bbox=(965, 405, 1250, 440),
    )

    page = DocumentPage(
        page_number=2,
        text=f"{block.text}\n\n{previous.text}",
        blocks=[block, previous],
        width=1440,
        height=810,
    )

    result = classify_block_role(
        block=block,
        page=page,
        dominant_font_size=21.0,
        previous=previous,
        following=None,
    )

    assert result["role"] == "heading"


def test_missing_following_block_has_zero_vertical_gap():
    block = DocumentBlock(
        text="Definition",
        page_number=1,
        font_size=21.0,
        is_bold=False,
        bbox=(40, 400, 200, 430),
    )

    page = make_page(block)

    result = get_normalized_relationship_features(
        block,
        None,
        page,
    )

    assert result["relative_vertical_gap"] == 0.0


def test_bold_short_text_is_classified_as_heading():
    block = DocumentBlock(
        text="Asset Tracking",
        page_number=2,
        font_size=21.0,
        is_bold=True,
        bbox=(40, 400, 250, 430),
    )

    page = make_page(block)

    result = classify_block_role(
        block=block,
        page=page,
        dominant_font_size=21.0,
        previous=None,
        following=None,
    )

    assert result["role"] == "heading"


def test_grouping_attaches_multiple_supporting_blocks():
    heading = DocumentBlock(
        text="Asset Management",
        page_number=2,
        font_size=26.0,
        is_bold=True,
        bbox=(40, 300, 300, 340),
    )

    body_one = DocumentBlock(
        text=(
            "Asset management helps organizations track and "
            "maintain physical resources efficiently across "
            "multiple operational environments."
        ),
        page_number=2,
        font_size=21.0,
        is_bold=False,
        bbox=(40, 380, 900, 450),
    )

    body_two = DocumentBlock(
        text=(
            "Digital systems provide visibility into ownership, "
            "location, lifecycle status, and maintenance activity."
        ),
        page_number=2,
        font_size=21.0,
        is_bold=False,
        bbox=(40, 470, 900, 540),
    )

    page = DocumentPage(
        page_number=2,
        text=(
            f"{heading.text}\n\n"
            f"{body_one.text}\n\n"
            f"{body_two.text}"
        ),
        blocks=[
            heading,
            body_one,
            body_two,
        ],
        width=1000,
        height=1000,
    )

    groups = group_blocks_by_layout(page)

    assert len(groups) == 1
    assert len(groups[0].blocks) == 3
    assert groups[0].blocks[0].text == heading.text


def test_detect_group_rows_groups_similar_vertical_positions():
    page = DocumentPage(
        page_number=1,
        text="",
        blocks=[],
        width=1000,
        height=1000,
    )

    groups = [
        VisualGroup(
            blocks=[],
            center_x=200,
            center_y=200,
            top_y=180,
            bbox=(100, 180, 300, 220),
        ),
        VisualGroup(
            blocks=[],
            center_x=700,
            center_y=210,
            top_y=190,
            bbox=(600, 190, 800, 230),
        ),
        VisualGroup(
            blocks=[],
            center_x=200,
            center_y=600,
            top_y=580,
            bbox=(100, 580, 300, 620),
        ),
    ]

    rows = detect_group_rows(
        page,
        groups,
    )

    assert len(rows) == 2
    assert len(rows[0]) == 2
    assert len(rows[1]) == 1


def test_detect_group_rows_sorts_groups_left_to_right():
    page = DocumentPage(
        page_number=1,
        text="",
        blocks=[],
        width=1000,
        height=1000,
    )

    groups = [
        VisualGroup(
            blocks=[],
            center_x=800,
            center_y=200,
            top_y=180,
            bbox=(700, 180, 900, 220),
        ),
        VisualGroup(
            blocks=[],
            center_x=200,
            center_y=210,
            top_y=190,
            bbox=(100, 190, 300, 230),
        ),
        VisualGroup(
            blocks=[],
            center_x=500,
            center_y=205,
            top_y=185,
            bbox=(400, 185, 600, 225),
        ),
    ]

    rows = detect_group_rows(
        page,
        groups,
    )

    assert len(rows) == 1
    assert [
        group.center_x
        for group in rows[0]
    ] == [200, 500, 800]


def test_detect_group_rows_sorts_rows_top_to_bottom():
    page = DocumentPage(
        page_number=1,
        text="",
        blocks=[],
        width=1000,
        height=1000,
    )

    groups = [
        VisualGroup(
            blocks=[],
            center_x=200,
            center_y=700,
            top_y=680,
            bbox=(100, 680, 300, 720),
        ),
        VisualGroup(
            blocks=[],
            center_x=200,
            center_y=200,
            top_y=180,
            bbox=(100, 180, 300, 220),
        ),
        VisualGroup(
            blocks=[],
            center_x=200,
            center_y=450,
            top_y=430,
            bbox=(100, 430, 300, 470),
        ),
    ]

    rows = detect_group_rows(
        page,
        groups,
    )

    assert len(rows) == 3
    assert [
        row[0].center_y
        for row in rows
    ] == [200, 450, 700]


def test_order_visual_groups_places_left_region_before_right_stack():
    page = DocumentPage(
        page_number=8,
        text="",
        blocks=[],
        width=1000,
        height=1000,
    )

    groups = [
        VisualGroup(
            blocks=[],
            center_x=800,
            center_y=500,
            top_y=450,
            bbox=(650, 450, 950, 550),
        ),
        VisualGroup(
            blocks=[],
            center_x=300,
            center_y=500,
            top_y=300,
            bbox=(50, 300, 550, 700),
        ),
        VisualGroup(
            blocks=[],
            center_x=800,
            center_y=750,
            top_y=700,
            bbox=(650, 700, 950, 800),
        ),
        VisualGroup(
            blocks=[],
            center_x=800,
            center_y=250,
            top_y=200,
            bbox=(650, 200, 950, 300),
        ),
    ]

    ordered = order_visual_groups(
        page,
        groups,
    )

    assert [
        group.center_x
        for group in ordered
    ] == [
        300,
        800,
        800,
        800,
    ]

    assert [
        group.center_y
        for group in ordered
    ] == [
        500,
        250,
        500,
        750,
    ]


def test_order_visual_groups_orders_simple_rows_left_to_right():
    page = DocumentPage(
        page_number=1,
        text="",
        blocks=[],
        width=1000,
        height=1000,
    )

    groups = [
        VisualGroup(
            blocks=[],
            center_x=800,
            center_y=200,
            top_y=180,
            bbox=(700, 180, 900, 220),
        ),
        VisualGroup(
            blocks=[],
            center_x=200,
            center_y=200,
            top_y=180,
            bbox=(100, 180, 300, 220),
        ),
        VisualGroup(
            blocks=[],
            center_x=500,
            center_y=200,
            top_y=180,
            bbox=(400, 180, 600, 220),
        ),
    ]

    ordered = order_visual_groups(
        page,
        groups,
    )

    assert [
        group.center_x
        for group in ordered
    ] == [
        200,
        500,
        800,
    ]


def test_order_visual_groups_orders_vertical_stack_top_to_bottom():
    page = DocumentPage(
        page_number=1,
        text="",
        blocks=[],
        width=1000,
        height=1000,
    )

    groups = [
        VisualGroup(
            blocks=[],
            center_x=800,
            center_y=700,
            top_y=680,
            bbox=(700, 680, 900, 720),
        ),
        VisualGroup(
            blocks=[],
            center_x=800,
            center_y=200,
            top_y=180,
            bbox=(700, 180, 900, 220),
        ),
        VisualGroup(
            blocks=[],
            center_x=800,
            center_y=450,
            top_y=430,
            bbox=(700, 430, 900, 470),
        ),
    ]

    ordered = order_visual_groups(
        page,
        groups,
    )

    assert [
        group.center_y
        for group in ordered
    ] == [
        200,
        450,
        700,
    ]