from datetime import datetime, timezone
import unittest

from bot import (
    Draft,
    Item,
    choose_publications,
    is_near_duplicate,
    normalize_url,
    render_post,
    safe_tag,
    stable_id,
    term_in_text,
)


def sample_item() -> Item:
    return Item(
        item_id="abc",
        source="Example & Source",
        source_kind="news",
        source_weight=2,
        title="New <AI> model",
        summary="A useful model.",
        url="https://example.com/post?a=1&utm_source=test",
        published_at=datetime.now(timezone.utc).isoformat(),
        score=8.0,
    )


class BotTests(unittest.TestCase):
    def test_normalize_url_removes_tracking_and_fragment(self):
        self.assertEqual(
            normalize_url("https://EXAMPLE.com/post/?a=1&utm_source=x#part"),
            "https://example.com/post?a=1",
        )

    def test_stable_id_ignores_tracking_parameters(self):
        first = stable_id("https://example.com/a?utm_source=x", "One")
        second = stable_id("https://example.com/a?utm_source=y", "Two")
        self.assertEqual(first, second)

    def test_similar_titles_are_duplicates(self):
        self.assertTrue(
            is_near_duplicate(
                "OpenAI launches a new reasoning model",
                ["OpenAI launches the new reasoning model"],
            )
        )

    def test_render_post_escapes_html_and_keeps_source(self):
        draft = Draft(
            item_id="abc",
            publish=True,
            score=8,
            category="news",
            headline="Модель <стала> доступна",
            facts=["Работает с A & B"],
            why_it_matters="Можно тестировать.",
            tags=["ИИ новости", "api"],
        )
        message = render_post(sample_item(), draft)
        self.assertIn("&lt;стала&gt;", message)
        self.assertIn("A &amp; B", message)
        self.assertIn('href="https://example.com/post?a=1&amp;utm_source=test"', message)
        self.assertIn("#ИИ_новости", message)

    def test_safe_tag_removes_punctuation(self):
        self.assertEqual(safe_tag("AI & bots!"), "AI_bots")

    def test_short_ai_term_matches_whole_word_only(self):
        self.assertTrue(term_in_text("ai", "a new ai model"))
        self.assertFalse(term_in_text("ai", "training platform"))

    def test_single_post_alternates_category(self):
        news = Draft("news", True, 7, "news", "News", [], "", [])
        guide = Draft("guide", True, 9, "guide", "Guide", [], "", [])
        news_item = sample_item()
        guide_item = sample_item()
        guide_item.item_id = "guide"
        chosen = choose_publications([(guide_item, guide), (news_item, news)], 1, "guide")
        self.assertEqual(chosen[0][1].category, "news")


if __name__ == "__main__":
    unittest.main()
