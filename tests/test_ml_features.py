from app.ml.features import word_overlap_ratio, length_ratio


def test_word_overlap_identical_strings():
    assert word_overlap_ratio("the cat sat", "the cat sat") == 1.0


def test_word_overlap_no_shared_words():
    assert word_overlap_ratio("apples oranges", "bananas grapes") == 0.0


def test_word_overlap_partial():
    result = word_overlap_ratio("the cat sat on mat", "the dog sat on mat")
    assert 0.0 < result < 1.0


def test_word_overlap_empty_string_returns_zero():
    assert word_overlap_ratio("", "something") == 0.0
    assert word_overlap_ratio("something", "") == 0.0


def test_length_ratio_equal_length():
    assert length_ratio("one two three", "four five six") == 1.0


def test_length_ratio_different_length():
    result = length_ratio("one two", "one two three four")
    assert result == 0.5


def test_length_ratio_empty_strings_returns_zero():
    assert length_ratio("", "") == 0.0