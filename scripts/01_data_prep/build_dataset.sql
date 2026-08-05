SET threads = 16;
SET memory_limit = '24GB'; 

SET preserve_insertion_order=false

INSTALL fts;
LOAD fts;

CREATE TABLE cleaned_table AS
SELECT
    * EXCLUDE (
        "ai_insight",
        "translation_type",
        "source_version",
        "document_hash",
        "group_hash",
        "group_id",
        "build_status",
        "file_metadata",
        "processed_at",
    ),
    STRUCT_PACK(law_insight := "ai_insight"."law_insight") AS ai_insight
FROM
    "data-duckdb".my_table;

CREATE OR REPLACE VIEW docs_view AS
SELECT
    build_id,
    meta.title AS title,
    classification.short_title AS short_title,
    classification.classification.document_type AS doc_type,
    meta.extra_data_json.document_url AS doc_url,
    ai_insight.law_insight.aliases AS aliases
FROM cleaned_table;

CREATE OR REPLACE TABLE aliases_view AS
SELECT
    build_id,
    meta.title AS title,
    classification.short_title AS short_title,
    alias
FROM cleaned_table
CROSS JOIN UNNEST(
    json_extract("ai_insight"."law_insight"."aliases", '$')::VARCHAR[]
) AS t(alias)
WHERE json_type("ai_insight"."law_insight"."aliases") = 'ARRAY';

CREATE OR REPLACE VIEW refs_view AS
SELECT
    build_id AS source_build_id,
    r.unnest.title AS ref_title,
    r.unnest.relationship_type AS relationship_type,
    r.unnest.description AS description,
FROM cleaned_table,
     unnest(ai_insight.law_insight.related_documents) AS r;

CREATE OR REPLACE MACRO clean_title(t) AS (
    trim(
        regexp_replace(
            regexp_replace(
                lower(t),
                '(commission|implementing|regulation|directive|decision|\(eu\)|[0-9]{4}/[0-9]+|of\s+[0-9]{1,2}\s+[a-z]+\s+[0-9]{4})',
                ' ',
                'gi'
            ),
            '\s+',
            ' ',
            'g'
        )
    )
);

CREATE OR REPLACE MACRO match_title(t) AS (
    trim(
        regexp_replace(
            regexp_replace(
                clean_title(t),
                '\b(imposing|a|an|the|of|on|in)\b',
                ' ',
                'gi'
            ),
            '\s+',
            ' ',
            'g'
        )
    )
);

CREATE OR REPLACE TEMP VIEW refs_clean AS
SELECT
    source_build_id,
    ref_title,
    match_title(ref_title) AS clean_title
FROM refs_view;


CREATE OR REPLACE TEMP VIEW docs_clean AS
SELECT
    build_id,
    title,
    match_title(title) AS clean_title
FROM docs_view;

CREATE OR REPLACE TEMP VIEW doc_words AS
SELECT
    build_id,
    lower(word) AS word
FROM docs_clean,
unnest(
    regexp_split_to_array(clean_title, '\s+')
) AS t(word)
WHERE length(word) > 3;


CREATE OR REPLACE TEMP VIEW word_frequency AS
SELECT
    word,
    count(DISTINCT build_id) AS docs_with_word
FROM doc_words
GROUP BY word;

CREATE OR REPLACE TEMP VIEW rare_words AS
SELECT
    word
FROM word_frequency
WHERE docs_with_word <= 50;

CREATE OR REPLACE TEMP VIEW docs_rare AS
SELECT
    d.build_id,
    d.title,
    rw.word
FROM docs_clean d
JOIN doc_words w
    ON d.build_id = w.build_id
JOIN rare_words rw
    ON w.word = rw.word;


CREATE OR REPLACE TEMP VIEW refs_rare AS
SELECT
    r.source_build_id,
    r.ref_title,
    rw.word
FROM refs_clean r
CROSS JOIN unnest(
    regexp_split_to_array(r.clean_title, '\s+')
) AS t(word)
JOIN rare_words rw
    ON lower(t.word) = rw.word;

CREATE OR REPLACE TEMP VIEW candidates AS
SELECT
    r.source_build_id,
    d.build_id,
    r.ref_title,
    d.title,

    count(*) AS rare_word_matches

FROM refs_rare r
JOIN docs_rare d
    ON r.word = d.word

GROUP BY
    r.source_build_id,
    d.build_id,
    r.ref_title,
    d.title

HAVING count(*) >= 1;

CREATE OR REPLACE TEMP VIEW scored AS
SELECT
    c.source_build_id,
    c.build_id,
    c.ref_title,
    c.title,

    c.rare_word_matches,

    jaro_winkler_similarity(
        match_title(c.ref_title),
        match_title(c.title)
    ) AS similarity,

    (
        c.rare_word_matches * 10
        +
        jaro_winkler_similarity(
            match_title(c.ref_title),
            match_title(c.title)
        ) * 100
    ) AS score

FROM candidates c;

CREATE OR REPLACE TEMP VIEW ranked_candidates AS
SELECT *
FROM (
    SELECT
        *,
        row_number() OVER (
            PARTITION BY source_build_id
            ORDER BY score DESC
        ) AS rank
    FROM scored
);

CREATE OR REPLACE VIEW docs_numbers AS
SELECT
    d.build_id,
    d.title,
    number AS doc_number
FROM docs_view d
CROSS JOIN UNNEST(
    extract_doc_numbers(d.title)
) AS t(number);

-- Extract referenced document identifiers from references
CREATE OR REPLACE VIEW refs_identifiers AS
SELECT
    r.source_build_id,
    r.ref_title,
    r.relationship_type,

    lower(
        regexp_extract(
            r.ref_title,
            '(?i)(regulation|directive|decision|opinion|recommendation)',
            1
        )
    ) AS title_extracted_ref_doc_type,

    list_sort(extract_doc_numbers(r.ref_title)) AS ref_doc_numbers,

    extract_ref_context(r.ref_title) AS ref_context

FROM refs_view r;

CREATE OR REPLACE VIEW docs_identifiers AS
SELECT
    d.build_id,
    d.title,

    lower(
        regexp_extract(
            d.title,
            '(?i)(regulation|directive|decision|opinion|recommendation)',
            1
        )
    ) AS title_extracted_doc_type,

    list_sort(extract_doc_numbers(d.title)) AS doc_numbers,

    extract_ref_context(d.title) AS doc_context,

    d.doc_type

FROM docs_view d;

CREATE OR REPLACE VIEW refs_identifiers_expanded AS
SELECT
    r.source_build_id,
    r.ref_title,
    r.relationship_type,

    lower(
        regexp_extract(
            r.ref_title,
            '(?i)(regulation|directive|decision|opinion|recommendation)',
            1
        )
    ) AS title_extracted_ref_doc_type,

    upper(t.identifier) AS ref_doc_number,

    t.ordinality AS ref_position,

    -- classify the context around THIS identifier occurrence
    extract_ref_context(
        substr(
            r.ref_title,
            greatest(
                1,
                strpos(
                    upper(r.ref_title),
                    upper(t.identifier)
                ) - 100
            ),
            150
        )
    ) AS ref_context

FROM refs_view r

CROSS JOIN UNNEST(
    extract_doc_numbers(r.ref_title)
)
WITH ORDINALITY AS t(identifier, ordinality);

CREATE OR REPLACE VIEW docs_identifiers_expanded AS
SELECT
    d.build_id,
    d.title,

    lower(
        regexp_extract(
            d.title,
            '(?i)(regulation|directive|decision|opinion|recommendation)',
            1
        )
    ) AS title_extracted_doc_type,

    upper(t.identifier) AS doc_number,

    t.ordinality AS doc_position,

    extract_ref_context(
        substr(
            d.title,
            greatest(
                1,
                strpos(
                    upper(d.title),
                    upper(t.identifier)
                ) - 100
            ),
            150
        )
    ) AS doc_context,

    d.doc_type

FROM docs_view d

CROSS JOIN UNNEST(
    extract_doc_numbers(d.title)
)
WITH ORDINALITY AS t(identifier, ordinality);

CREATE OR REPLACE VIEW identifier_exact_matches AS
SELECT
    source_build_id,
    build_id,

    ref_title,
    title,

    ref_doc_numbers,
    doc_numbers,

    ref_context,
    doc_context,

    strategy,

    score

FROM (

    SELECT
        r.source_build_id,
        d.build_id,

        r.ref_title,
        d.title,

        r.ref_doc_numbers,
        d.doc_numbers,

        r.ref_context,
        d.doc_context,

        'identifier_exact' AS strategy,

        CASE

            -- strongest: title identifies same document
            WHEN r.ref_context = d.doc_context
             AND r.ref_doc_numbers = d.doc_numbers
            THEN 100


            -- reference relationship to another document
            WHEN r.ref_context IN (
                    'amending',
                    'repealing',
                    'implementing'
                 )
             AND d.doc_context = 'identity'
             AND len(list_intersect(
                    r.ref_doc_numbers,
                    d.doc_numbers
                 )) > 0
            THEN 40


            -- legal basis citation
            WHEN r.ref_context = 'legal_basis'
             AND d.doc_context = 'identity'
             AND len(list_intersect(
                    r.ref_doc_numbers,
                    d.doc_numbers
                 )) > 0
            THEN 20

            ELSE 0

        END AS score

    FROM refs_identifiers r

    JOIN docs_identifiers d

        ON (
            -- exact identity match
            (
                r.ref_context = 'identity'
                AND r.ref_doc_numbers = d.doc_numbers
            )

            OR

            -- contextual references
            (
                r.ref_context <> 'identity'
                AND len(list_intersect(
                    r.ref_doc_numbers,
                    d.doc_numbers
                )) > 0
            )
        )
    WHERE array_length(r.ref_doc_numbers) > 0
)

WHERE score > 0

QUALIFY ROW_NUMBER() OVER (
    PARTITION BY source_build_id, build_id
    ORDER BY score DESC
);


CREATE OR REPLACE MACRO extract_doc_numbers(t) AS (
    regexp_extract_all(
        upper(t),
        '('
        -- 1. Standard numeric document citations (e.g. 2025/1135, 12/2024)
        || '\b[0-9]{1,5}/[0-9]{2,4}\b'
        
        -- 2. Placeholder patterns (e.g. …/…, .../..., 202X/XXXX, ____/____)
        || '|…/…'
        || '|\.\.\./\.\.\.'
        || '|\b[0-9X_]{2,4}/[0-9X_]{2,4}\b'
        || ')'
    )
);

CREATE OR REPLACE MACRO extract_ref_context(t) AS (
    CASE

        -- Changes / amendments to another act
        WHEN regexp_matches(
            lower(t),
            '(amending|amend|amended|modifying|modified|modifies|modifiant|modifiant le|modifie|modifiant le règlement|'
            || 'supplementing|supplemented|supplements|supplementant|complétant|complétant le|'
            || 'adding to|altering|adaptant|adaptation)'
        )
        THEN 'amending'


        -- Repeal / replacement
        WHEN regexp_matches(
            lower(t),
            '(repealing|repealed|repeal|replaces|replaced|replacement|replacing|'
            || 'abrogating|abrogeant|abroge|aufhebend|aufhebung|derogating|deroga)'
        )
        THEN 'repealing'


        -- Implementing acts
        WHEN regexp_matches(
            lower(t),
            '(implementing|implement|implementation|implementing regulation|'
            || 'implementing decision|exécution|d’exécution|execution|'
            || 'zur durchführung|zur Durchführung|'
            || 'de ejecución|ejecución)'
        )
        THEN 'implementing'


        -- Delegated acts
        WHEN regexp_matches(
            lower(t),
            '(delegated|delegating|delegated regulation|delegated act|'
            || 'délégué|déléguée|déléguée par|'
            || 'delegierte|delegierter)'
        )
        THEN 'delegated'


        -- Corrections / corrigenda
        WHEN regexp_matches(
            lower(t),
            '(correcting|corrected|correction|corrigendum|corrigenda|'
            || 'rectifiant|rectificatif|rectification|'
            || 'berichtigung|berichtigend)'
        )
        THEN 'correcting'


        -- Legal basis / authority citation
        WHEN regexp_matches(
            lower(t),
            '(in accordance with|pursuant to|having regard to|'
            || 'on the basis of|under|by virtue of|'
            || 'vu le|vu la|conformément à|en vertu de|'
            || 'gestützt auf|nach maßgabe|'
            || 'de conformidad con|con arreglo a)'
        )
        THEN 'legal_basis'


        -- References to another act without modification
        WHEN regexp_matches(
            lower(t),
            '(following|according to|'
            || 'as regards|with regard to|concerning|relating to|'
            || 'as provided for in|'
            || 'en ce qui concerne|concernant|'
            || 'betreffend|hinsichtlich|'
            || 'por lo que respecta a|'
            || 'covered by the provisions of|'
            || 'falling within the scope of|'
            || 'within the scope of|'
            || 'subject to|'
            || 'governed by|'
            || 'regulated by)'
        )
        THEN 'reference'

        -- Drafts, Proposals, and Preparatory Acts
        WHEN regexp_matches(
            lower(t),
            '('
            -- Basic keywords & variations
            || 'proposal|proposition|vorschlag|propuesta|proposta|proposta di|'
            -- Draft terms
            || 'draft|projet|entwurf|proyecto|progetto|'
            -- Preparatory & Working Documents
            || 'working document|document de travail|arbeitsdokument|documento de trabajo|'
            || 'staff working|preparatory act|acte préparatoire|vorarbeit|'
            -- Institutional stage markers (EU specific)
            || 'white paper|green paper|livre vert|livre blanc|grünbuch|weißbuch|'
            || 'initiative|opinion of the commission|avis de la commission|'
            -- Common title constructs / prefix patterns
            || 'for a regulation|for a directive|for a decision|for a council|'
            || 'd''un règlement|d''une directive|para un reglamento|para una directiva|'
            -- Unassigned / Placeholder indicators
            || '…/\.\.\.|…/…|\.\.\./\.\.\.|20[0-9]{2}/___|20[0-9]{2}/xxxx'
            || ')'
        )
        THEN 'proposal'

        ELSE 'identity'

    END
);

CREATE OR REPLACE VIEW identifier_exact_matches AS
WITH ref_sets AS (

    SELECT
        source_build_id,
        ref_title,

        list_sort(
            list(
                struct_pack(
                    ctx := ref_context,
                    num := ref_doc_number
                )
            )
        ) AS ref_tuples

    FROM refs_identifiers_expanded

    GROUP BY
        source_build_id,
        ref_title

),

doc_sets AS (

    SELECT
        build_id,
        title,

        list_sort(
            list(
                struct_pack(
                    ctx := doc_context,
                    num := doc_number
                )
            )
        ) AS doc_tuples

    FROM docs_identifiers_expanded

    GROUP BY
        build_id,
        title

)

SELECT

    r.source_build_id,
    d.build_id,

    r.ref_title,
    d.title,

    r.ref_tuples,
    d.doc_tuples,

    'identifier_exact' AS strategy,

    100 AS score

FROM ref_sets r

JOIN doc_sets d
    ON r.ref_tuples = d.doc_tuples;

    
CREATE OR REPLACE VIEW matches AS
WITH matches AS (
    -- Strategy 1: exact title
    SELECT 
        r.source_build_id,
        d.build_id,
        r.ref_title,
        d.title,
        'title_exact' AS strategy
    FROM refs_view r
    JOIN docs_view d 
        ON lower(d.title) = lower(r.ref_title)

    UNION ALL

    -- Strategy 2: short title exact
    SELECT 
        r.source_build_id,
        d.build_id,
        r.ref_title,
        d.title,
        'title_is_short_title' AS strategy
    FROM refs_view r
    JOIN docs_view d 
        ON lower(d.short_title) = lower(r.ref_title)

    UNION ALL

    -- Strategy 3: short title prefix
    SELECT 
        r.source_build_id,
        d.build_id,
        r.ref_title,
        d.title,
        'short_title_prefix' AS strategy
    FROM refs_view r
    JOIN docs_view d 
        ON r.ref_title ILIKE d.short_title || '%'

    UNION ALL

    -- Strategy 4: alias exact
    SELECT 
        r.source_build_id,
        d.build_id,
        r.ref_title,
        d.title,
        'alias_exact' AS strategy
    FROM refs_view r
    JOIN aliases_view a 
        ON a.alias = r.ref_title
    JOIN docs_view d
        ON d.build_id = a.build_id

    UNION ALL

    -- Strategy 5: extracted identifier match
    SELECT
        iem.source_build_id,
        iem.build_id,
        iem.ref_title,
        iem.title,
        'extracted_identifier_matcher' AS strategy
    FROM identifier_exact_matches iem
) SELECT * FROM matches;

COPY (
    WITH dedup_matches AS (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY build_id, source_build_id
                   ORDER BY build_id, ref_title
               ) AS rn
        FROM matches
    )
    SELECT json_object(
        'reference',
        json_object(
            'source_build_id', r.source_build_id,
            'ref_title', r.ref_title,
            'relationship_type', r.relationship_type,
            'description', r.description
        ),
        'positive',
        json_object(
            'build_id', d.build_id,
            'title', d.title,
            'short_title', d.short_title,
            'doc_type', d.doc_type,
            'doc_url', d.doc_url,
            'aliases', d.aliases
        ),
        'strategy', m.strategy
    ) AS json
    FROM refs_view r
    JOIN dedup_matches m
        ON r.source_build_id = m.source_build_id
       AND r.ref_title = m.ref_title
    JOIN docs_view d
        ON d.build_id = m.build_id
    WHERE m.rn = 1
) TO 'data/positive_pairs.jsonl'
(FORMAT JSON);

COPY (
    SELECT * FROM docs_view
) TO 'data/docs.jsonl';

COPY (
    SELECT * FROM refs_view
) TO 'data/refs.jsonl';