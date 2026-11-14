<?php
declare(strict_types=1);

/**
 * Minimal BERT WordPiece tokenizer for the BGE-small benchmark.
 *
 * This is intentionally limited to the tokenizer.json shape used by the
 * benchmark model: BertNormalizer (lowercase), BertPreTokenizer, WordPiece
 * with "##" continuation prefixes, and [CLS]/[SEP] template processing.
 */
final class BenchWordPieceTokenizer
{
    /** @var array<string, int> */
    private array $vocab;
    private string $unkToken;
    private string $prefix;
    private int $maxInputChars;

    public function __construct(string $tokenizerJsonPath)
    {
        $json = json_decode(file_get_contents($tokenizerJsonPath), true, 512, JSON_THROW_ON_ERROR);
        $model = $json['model'] ?? null;
        if (!is_array($model) || ($model['type'] ?? '') !== 'WordPiece') {
            throw new RuntimeException('expected a WordPiece tokenizer.json');
        }
        $this->vocab = $model['vocab'];
        $this->unkToken = $model['unk_token'] ?? '[UNK]';
        $this->prefix = $model['continuing_subword_prefix'] ?? '##';
        $this->maxInputChars = (int) ($model['max_input_chars_per_word'] ?? 100);
    }

    /** @return array{input_ids: list<list<int>>, attention_mask: list<list<int>>, token_type_ids: list<list<int>>} */
    public function encodeBatch(string $text, int $batchSize, int $sequenceLength): array
    {
        $ids = $this->encode($text, $sequenceLength);
        $attention = array_fill(0, count($ids), 1);
        for ($i = count($ids); $i < $sequenceLength; $i++) {
            $ids[] = (int) ($this->vocab['[PAD]'] ?? 0);
            $attention[] = 0;
        }
        $types = array_fill(0, $sequenceLength, 0);

        $batchIds = [];
        $batchAttention = [];
        $batchTypes = [];
        for ($i = 0; $i < $batchSize; $i++) {
            $batchIds[] = $ids;
            $batchAttention[] = $attention;
            $batchTypes[] = $types;
        }
        return [
            'input_ids' => $batchIds,
            'attention_mask' => $batchAttention,
            'token_type_ids' => $batchTypes,
        ];
    }

    /** @return list<int> */
    private function encode(string $text, int $sequenceLength): array
    {
        $text = $this->normalize($text);
        preg_match_all('/[\p{L}\p{N}]+|[^\p{L}\p{N}\s]+/u', $text, $matches);
        $words = $matches[0] ?? [];

        $ids = [(int) ($this->vocab['[CLS]'] ?? 101)];
        $budget = max(0, $sequenceLength - 2);
        foreach ($words as $word) {
            if (count($ids) - 1 >= $budget) {
                break;
            }
            foreach ($this->wordPiece($word) as $token) {
                if (count($ids) - 1 >= $budget) {
                    break 2;
                }
                $ids[] = $token;
            }
        }
        $ids[] = (int) ($this->vocab['[SEP]'] ?? 102);
        return $ids;
    }

    private function normalize(string $text): string
    {
        $text = preg_replace('/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/u', '', $text) ?? $text;
        $text = preg_replace('/\s+/u', ' ', $text) ?? $text;
        $text = trim($text);
        return mb_strtolower($text, 'UTF-8');
    }

    /** @return list<int> */
    private function wordPiece(string $word): array
    {
        $chars = preg_split('//u', $word, -1, PREG_SPLIT_NO_EMPTY) ?: [];
        $length = count($chars);
        if ($length === 0) {
            return [];
        }
        if ($length > $this->maxInputChars) {
            return [(int) ($this->vocab[$this->unkToken] ?? 100)];
        }

        $tokens = [];
        $start = 0;
        while ($start < $length) {
            $end = $length;
            $matched = null;
            while ($end > $start) {
                $piece = implode('', array_slice($chars, $start, $end - $start));
                $candidate = $start === 0 ? $piece : $this->prefix . $piece;
                if (isset($this->vocab[$candidate])) {
                    $matched = [$candidate, $end];
                    break;
                }
                $end--;
            }
            if ($matched === null) {
                return [(int) ($this->vocab[$this->unkToken] ?? 100)];
            }
            $tokens[] = (int) $this->vocab[$matched[0]];
            $start = $matched[1];
        }
        return $tokens;
    }
}
