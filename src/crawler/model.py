from dataclasses import dataclass


@dataclass(slots=True)
class TikiReview:
    review_id: int | None
    product_id: int
    product_name: str | None
    customer_id: int | None
    customer_name: str | None
    rating: int | None
    title: str | None
    content: str | None
    thank_count: int | None
    comment_count: int | None
    created_at: str | None
    is_photo: bool
    seller_name: str | None
    category_id: str | None = None
    category_name: str | None = None
    source: str = "tiki"
    crawled_at: str | None = None


@dataclass(slots=True)
class ProductSeed:
    source: str
    category_id: str
    category_name: str
    product_id: str
    product_name: str
    product_url: str
    review_count: int = 0
    quantity_sold: int = 0
    rating_average: float = 0.0
    price: float = 0.0
    original_price: float = 0.0
    status: str = "pending"
    crawled_at: str | None = None

@dataclass(slots=True)
class LazadaReview:
    review_id: str | int | None
    product_id: str | int
    product_name: str | None
    customer_id: str | int | None
    customer_name: str | None
    rating: int | None
    title: str | None
    content: str | None
    thank_count: int | None = 0
    comment_count: int | None = 0
    created_at: str | None = None
    is_photo: bool = False
    seller_name: str | None = None
    category_id: str | None = None
    category_name: str | None = None
    source: str = "lazada"
    crawled_at: str | None = None


Review = TikiReview | LazadaReview


class TikiCrawlerError(RuntimeError):
    pass


class LazadaCrawlerError(RuntimeError):
    pass
