from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .models import Confidence, SectionType
from .rules import SECTION_RULES, SectionRule, HEADER_SUFFIX_PATTERN
from .normalizer import get_lines, is_empty_line, is_bullet_line


@dataclass
class SectionAnchor:
    section_type: SectionType
    line_number: int
    line_text: str
    score: int
    confidence: Confidence
    matched_keyword: Optional[str] = None

    def __repr__(self) -> str:
        return f"Anchor({self.section_type.value}, line={self.line_number}, score={self.score})"


HIGH_CONFIDENCE_THRESHOLD = 60
MEDIUM_CONFIDENCE_THRESHOLD = 35


def detect_section_anchors(normalized_text: str) -> List[SectionAnchor]:
    lines = get_lines(normalized_text)
    anchors: List[SectionAnchor] = []
    total_lines = len(lines)

    for line_num, line in enumerate(lines):
        if is_empty_line(line) or is_bullet_line(line):
            continue

        scores = _calculate_line_scores(
            line=line,
            line_num=line_num,
            total_lines=total_lines,
            prev_empty=_is_prev_empty(lines, line_num),
            next_empty=_is_next_empty(lines, line_num),
        )

        best_match = _find_best_match(scores)
        if best_match:
            section_type, score, keyword = best_match
            confidence = _score_to_confidence(score)

            anchors.append(
                SectionAnchor(
                    section_type=section_type,
                    line_number=line_num,
                    line_text=line.strip(),
                    score=score,
                    confidence=confidence,
                    matched_keyword=keyword,
                )
            )

    anchors = _resolve_conflicts(anchors)

    return sorted(anchors, key=lambda a: a.line_number)


def _calculate_line_scores(
    line: str,
    line_num: int,
    total_lines: int,
    prev_empty: bool,
    next_empty: bool,
) -> Dict[SectionType, Tuple[int, Optional[str]]]:
    scores: Dict[SectionType, Tuple[int, Optional[str]]] = {}
    line_lower = line.lower().strip()

    line_clean = HEADER_SUFFIX_PATTERN.sub("", line_lower).strip()
    has_colon = line_lower.endswith(":") or line_lower.endswith("：")

    for section_type, rule in SECTION_RULES.items():
        score, keyword = _score_line_against_rule(
            line_clean=line_clean,
            line_lower=line_lower,
            rule=rule,
        )

        if score > 0:
            if has_colon:
                score += rule.colon_bonus

            if prev_empty and next_empty:
                score += rule.standalone_line_bonus
            elif prev_empty or next_empty:
                score += rule.standalone_line_bonus // 2

            position_ratio = line_num / max(total_lines, 1)
            if rule.position_score_multiplier != 1.0:
                if rule.position_score_multiplier > 1.0 and position_ratio < 0.3:
                    score = int(score * rule.position_score_multiplier)
                elif rule.position_score_multiplier < 1.0 and position_ratio > 0.7:
                    score = int(score * (2 - rule.position_score_multiplier))

            if len(line_clean) < 40:
                score += 5

            scores[section_type] = (score, keyword)

    return scores


def _score_line_against_rule(
    line_clean: str,
    line_lower: str,
    rule: SectionRule,
) -> Tuple[int, Optional[str]]:
    score = 0
    matched_keyword: Optional[str] = None

    for keyword in rule.keywords_ru:
        if _keyword_matches(line_clean, keyword):
            if len(keyword) > 3 and keyword in line_clean:
                score = rule.keyword_primary_score + 10
            else:
                score = rule.keyword_primary_score
            matched_keyword = keyword
            break

    if score == 0:
        for keyword in rule.keywords_en:
            if _keyword_matches(line_clean, keyword):
                if len(keyword) > 3 and keyword in line_clean:
                    score = rule.keyword_primary_score + 10
                else:
                    score = rule.keyword_primary_score
                matched_keyword = keyword
                break

    if score == 0:
        for keyword in rule.keywords_alt:
            if _keyword_matches(line_clean, keyword):
                score = rule.keyword_alt_score
                matched_keyword = keyword
                break

    return score, matched_keyword


def _keyword_matches(line: str, keyword: str) -> bool:
    keyword_lower = keyword.lower()
    if line == keyword_lower:
        return True

    if line.startswith(keyword_lower):
        remaining = line[len(keyword_lower) :]
        if not remaining or not remaining[0].isalnum():
            return True

    if " " in keyword_lower and keyword_lower in line:
        return True

    if " " not in keyword_lower:
        words = line.split()
        if keyword_lower in words:
            return True

    return False


def _is_prev_empty(lines: List[str], line_num: int) -> bool:
    if line_num == 0:
        return True
    return is_empty_line(lines[line_num - 1])


def _is_next_empty(lines: List[str], line_num: int) -> bool:
    if line_num >= len(lines) - 1:
        return True
    return is_empty_line(lines[line_num + 1])


def _find_best_match(
    scores: Dict[SectionType, Tuple[int, Optional[str]]],
) -> Optional[Tuple[SectionType, int, Optional[str]]]:
    if not scores:
        return None

    best_type = max(scores.keys(), key=lambda t: scores[t][0])
    best_score, best_keyword = scores[best_type]

    if best_score < MEDIUM_CONFIDENCE_THRESHOLD:
        return None

    return best_type, best_score, best_keyword


def _score_to_confidence(score: int) -> Confidence:
    if score >= HIGH_CONFIDENCE_THRESHOLD:
        return Confidence.HIGH
    elif score >= MEDIUM_CONFIDENCE_THRESHOLD:
        return Confidence.MEDIUM
    else:
        return Confidence.LOW


def _resolve_conflicts(anchors: List[SectionAnchor]) -> List[SectionAnchor]:
    if len(anchors) <= 1:
        return anchors

    type_to_best: Dict[SectionType, SectionAnchor] = {}
    for anchor in anchors:
        existing = type_to_best.get(anchor.section_type)
        if existing is None or anchor.score > existing.score:
            type_to_best[anchor.section_type] = anchor

    resolved = list(type_to_best.values())
    resolved.sort(key=lambda a: a.line_number)

    final: List[SectionAnchor] = []
    for anchor in resolved:
        if final and abs(anchor.line_number - final[-1].line_number) <= 2:
            if anchor.score > final[-1].score:
                final[-1] = anchor
        else:
            final.append(anchor)

    return final


def get_section_boundaries(
    anchors: List[SectionAnchor],
    total_lines: int,
) -> List[Tuple[SectionAnchor, int, int]]:
    if not anchors:
        return []

    boundaries: List[Tuple[SectionAnchor, int, int]] = []

    for i, anchor in enumerate(anchors):
        start_line = anchor.line_number
        if i + 1 < len(anchors):
            end_line = anchors[i + 1].line_number - 1
        else:
            end_line = total_lines - 1

        boundaries.append((anchor, start_line, end_line))

    return boundaries
