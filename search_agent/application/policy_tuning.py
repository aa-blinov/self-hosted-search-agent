from __future__ import annotations

# Claim profile defaults
DEFAULT_FACT_MIN_INDEPENDENT_SOURCES = 1
DEFAULT_OVERVIEW_MIN_INDEPENDENT_SOURCES = 2
DEFAULT_NEWS_DIGEST_MIN_INDEPENDENT_SOURCES = 3
PREFERRED_DOMAIN_TYPE_BONUS = 0.08

# Synthesis passage selection
NEWS_DIGEST_PRIMARY_PASSAGES_PER_URL = 1
NEWS_DIGEST_PRIMARY_PASSAGES_PER_DOMAIN = 1
DEFAULT_PRIMARY_PASSAGES_PER_URL = 1
DEFAULT_PRIMARY_PASSAGES_PER_DOMAIN = 999
NEWS_DIGEST_REPEAT_PASSAGES_PER_URL = 1
DEFAULT_REPEAT_PASSAGES_PER_URL = 2
NEWS_DIGEST_REPEAT_PASSAGES_PER_DOMAIN = 1
DEFAULT_REPEAT_PASSAGES_PER_DOMAIN = 999
NEWS_DIGEST_MIN_SYNTHESIS_SOURCES = 2
NEWS_DIGEST_MIN_FETCHED_URLS = 2

# Claim loop / verification policy
SUPPORTED_STOP_CONFIDENCE = 0.75
STRICT_EXACT_DETAIL_STOP_CONFIDENCE = 0.9
STRICT_EXACT_DETAIL_STOP_MIN_SOURCES = 2
SUPPORTED_CONFIDENCE_FLOOR_THRESHOLD = 0.05
SUPPORTED_CONFIDENCE_FLOOR_VALUE = 0.38
OPEN_ENDED_DEMOTION_CONFIDENCE_CAP = 0.62
OVERVIEW_ROBUST_PASSAGE_UTILITY_THRESHOLD = 0.2
OVERVIEW_ROBUST_PASSAGE_SOURCE_THRESHOLD = 0.7
OVERVIEW_SUPPORTING_PASSAGE_COUNT = 2
OVERVIEW_SUPPORTING_MIN_UNIQUE_HOSTS = 1
OVERVIEW_SUPPORTING_SPAN_CHARS = 220
OVERVIEW_PROMOTED_CONFIDENCE_FLOOR = 0.62

# SERP gating / source scoring
SPAM_SUFFIX_PENALTY = 0.6
SPAM_CUE_PENALTY = 0.25
SPAM_TITLE_SEPARATOR_PENALTY = 0.2
SPAM_LONG_TITLE_PENALTY = 0.1
SPAM_AUTHORITATIVE_DOMAIN_DISCOUNT = 0.2

PRIMARY_SOURCE_BASE_BY_DOMAIN_TYPE = {
    "official": 0.9,
    "academic": 0.85,
    "vendor": 0.7,
    "major_media": 0.45,
    "forum": 0.15,
    "unknown": 0.4,
}
PRIMARY_SOURCE_CUE_BOOST = 0.1
PRIMARY_SOURCE_ENTITY_HOST_MATCH_WEIGHT = 0.18
PRIMARY_SOURCE_PATH_CUE_BOOST = 0.08
PRIMARY_SOURCE_FORUM_PENALTY = 0.12

SERP_DOMAIN_PRIOR_BY_TYPE = {
    "official": 1.0,
    "academic": 0.95,
    "vendor": 0.75,
    "major_media": 0.7,
    "forum": 0.3,
    "unknown": 0.5,
}
SERP_DOMAIN_PRIOR_WEIGHT = 0.22
SERP_PRIMARY_WEIGHT = 0.22
SERP_FRESHNESS_WEIGHT = 0.12
SERP_ENTITY_MATCH_WEIGHT = 0.16
SERP_SEMANTIC_MATCH_WEIGHT = 0.12
SERP_HOST_ENTITY_WEIGHT = 0.06
SERP_TIME_ALIGNMENT_WEIGHT = 0.10
SERP_SPAM_PENALTY_WEIGHT = 0.25

VERIFICATION_BONUS_BY_DOMAIN_TYPE = {
    "official": 0.38,
    "academic": 0.34,
    "vendor": 0.18,
    "major_media": 0.08,
    "forum": -0.20,
    "unknown": 0.0,
}
VERIFICATION_RELEASE_CUE_BOOST = 0.14
VERIFICATION_FORUM_CUE_PENALTY = 0.12

# Passage scoring
CHEAP_PASSAGE_PRODUCT_WEIGHTS = {
    "overlap": 0.28,
    "entity_overlap": 0.20,
    "dimension_overlap": 0.16,
    "focus_overlap": 0.16,
    "source_score": 0.10,
    "preferred_bonus": 0.10,
}
CHEAP_PASSAGE_DEFAULT_WEIGHTS = {
    "overlap": 0.30,
    "entity_overlap": 0.22,
    "dimension_overlap": 0.18,
    "focus_overlap": 0.15,
    "number_overlap": 0.07,
    "source_score": 0.08,
}
UTILITY_NEWS_WEIGHTS = {
    "cheap_score": 0.24,
    "region_match": 0.24,
    "time_match": 0.18,
    "focus_overlap": 0.16,
    "source_bonus": 0.10,
    "source_score": 0.08,
}
UTILITY_PRODUCT_WEIGHTS = {
    "cheap_score": 0.36,
    "focus_overlap": 0.18,
    "preferred_bonus": 0.16,
    "source_bonus": 0.18,
    "source_score": 0.12,
}
UTILITY_DEFAULT_WEIGHTS = {
    "cheap_score": 0.30,
    "dimension_coverage": 0.20,
    "focus_overlap": 0.15,
    "directness": 0.15,
    "source_bonus": 0.12,
    "source_score": 0.08,
    "contradiction_signal": 0.05,
}
DIRECTNESS_TIME_BOOST = 0.3
DIRECTNESS_NUMBER_BOOST = 0.3
DIRECTNESS_PERSON_BOOST = 0.2
DIRECTNESS_LOCATION_BOOST = 0.2
CONTRADICTION_SIGNAL_BOOST = 0.15

SHALLOW_DOCUMENT_SCORE_WEIGHTS = {
    "semantic_overlap": 0.30,
    "entity_overlap": 0.22,
    "dimension_coverage": 0.18,
    "source_score": 0.18,
    "source_bonus": 0.12,
}
DIMENSION_COVERAGE = {
    "time_base": 0.35,
    "time_focus_weight": 0.20,
    "number_base_with_focus": 0.35,
    "number_base_without_focus": 0.05,
    "number_focus_weight": 0.25,
    "generic_focus_weight": 0.35,
    "time_scope_boost": 0.15,
    "person_boost": 0.20,
    "location_boost": 0.20,
    "number_focus_overlap_threshold": 0.25,
}
ROUTING = {
    "top_results_limit": 5,
    "candidate_consensus_limit": 3,
    "gated_results_support_window": 8,
    "evidence_score_cap": 0.45,
    "evidence_source_weight": 0.11,
    "evidence_source_threshold": 0.6,
    "evidence_primary_boost": 0.20,
    "evidence_primary_threshold": 0.7,
    "evidence_entity_boost": 0.15,
    "evidence_entity_threshold": 0.7,
    "evidence_semantic_boost": 0.15,
    "evidence_semantic_threshold": 0.7,
    "evidence_detail_weight": 0.15,
    "evidence_focus_weight": 0.12,
    "consistency_dimension_weight": 0.6,
    "exact_number_targeted_threshold": 0.4,
    "default_number_targeted_threshold": 0.45,
    "exact_detail_min_coverage": 0.45,
    "exact_detail_min_focus_overlap": 0.6,
    "short_path_certainty": 0.8,
    "short_path_consistency": 0.35,
    "short_path_sufficiency": 0.6,
    "targeted_number_min_coverage": 0.25,
    "targeted_number_min_sufficiency": 0.35,
    "targeted_time_certainty": 0.45,
    "targeted_default_certainty": 0.5,
    "targeted_default_consistency": 0.5,
    "targeted_default_sufficiency": 0.5,
    "targeted_fallback_certainty": 0.55,
    "targeted_fallback_sufficiency": 0.45,
}
CHEAP_PASSAGE_NEWS_WEIGHTS = {
    "overlap": 0.18,
    "region_match": 0.28,
    "time_match": 0.22,
    "focus_overlap": 0.16,
    "source_score": 0.16,
}
NEWS_SELECTION = {
    "min_region_match": 0.25,
    "region_weight": 0.42,
    "time_weight": 0.25,
    "local_bonus_weight": 0.15,
    "utility_or_source_weight": 0.18,
}
ANSWER_SENTENCE = {
    "entity_weight": 0.15,
    "time_specificity_weight": 0.65,
    "number_boost": 0.35,
    "person_boost": 0.20,
    "location_boost": 0.20,
    "time_scope_boost": 0.10,
}

SUPPORT_SCORE_WEIGHTS = {
    "semantic_overlap": 0.45,
    "entity_overlap": 0.20,
    "dimension_coverage": 0.20,
    "verification_bonus": 0.08,
    "utility_or_source": 0.07,
}
SUPPORT_SCORE_PRODUCT_PREFERRED_BONUS = 0.18
SUPPORT_SELECTION_WEIGHTS = {
    "semantic_overlap": 0.36,
    "entity_overlap": 0.16,
    "dimension_coverage": 0.16,
    "utility_or_source": 0.16,
    "verification_bonus": 0.10,
    "preferred_bonus": 0.06,
}
SUPPORT_SELECTION_PRODUCT_PREFERRED_BONUS = 0.14
