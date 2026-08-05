COPY(
    SELECT *
    FROM refs_view r
    WHERE r.source_build_id NOT IN (
        SELECT json.reference.source_build_id
        FROM read_json_auto('data/positive_pairs.jsonl')
    )
) TO 'data/unlabeled_refs.jsonl'

CREATE OR REPLACE VIEW refs_view_with_articles AS
SELECT
    build_id AS source_build_id,
    r.unnest.title AS ref_title,
    r.unnest.relationship_type AS relationship_type,
    r.unnest.description AS description,
    r.unnest.articles AS articles
FROM cleaned_table,
     unnest(ai_insight.law_insight.related_documents) AS r;

COPY (
    SELECT * FROM refs_view_with_articles
) TO 'data/refs_with_articles.jsonl'