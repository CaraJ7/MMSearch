import logging
import sys
import unicodedata
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize as sent_tok

from retrieve_content.tokenization.utils import PickleWriteable

QUOTES = re.compile("(\"|``|'')")

#Inherit dict to support to json
class TokenWithOffsets(dict):

    def __init__(self, token, start_offset, end_offset):
        if start_offset > end_offset:
            raise ValueError("start offset cannot be greater than end offset for a token")
        dict.__init__(self, _token=token, _start_offset=start_offset, _end_offset=end_offset)
        # token should only be changed by set_token() method
        # start_offset should not be changed by any external call
        # end_offset should not be changed by any external call

    def get_token(self):
        return self['_token']

    def set_token(self, tkn):
        self['_token'] = tkn

    def get_start_offset(self):
        return self['_start_offset']

    def get_end_offset(self):
        return self['_end_offset']


class WordTokenizer(PickleWriteable):
    """Class which tokenizes text at the word level."""

    PUNCTUATION_TABLE = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
    PUNCTUATION_CHARS = set(chr(i) for i in PUNCTUATION_TABLE.keys())

    #  The following are the arguments that can be used to initialize the WordTokenizer

    MIN_WORD_LEN = "min_word_len"  # min char length of an acceptable word
    MAX_WORD_LEN = "max_word_len"  # if a tokenized word exceeds this length, we will split it into shorter tokens
    # (can happen with foreign language text)
    SPACE_CHARS = "space_chars"  # all the characters that can be used as spaces
    DO_STOP_WORDS = "do_stop_words"  # do stop-word removal
    STOP_WORDS = "stop_words"  # stop-words list
    PASSAGE_LEN = "passage_len"  # maximum length of the passage (only approximately complied when
    MAX_SENT_LEN = "max_sent_len"  # maximum length of a sentence beyond which it will be broken up
    # respect_sent_boundaries is True
    RESPECT_SENT_BOUNDARIES = "respect_sent_boundaries"  # whether to respect sentence boundaries when splitting text
    # into passages
    DO_SLIDING_WINDOW_PASSAGES = "do_sliding_window_passages"  # split the text into overlapping passages based on
    # sliding windows. The stride of sliding window is set to half the passage length.
    DO_CHAR_OFFSETS = "do_char_offsets"  # return character offsets of the tokenized text with respect to the raw text
    STRIP_INWORD_PUNCTUATION = "strip_inword_punctuation"  # break the words at punctuation characters in the middle of
    # words. e.g.: can't --> [ca, nt]
    REMOVE_NUMERICS = "remove_numerics"  # filter out numerical values from text
    DO_STEM = "do_stem"  # apply stemming to the tokens
    DO_LEMMA = "do_lemma"  # apply lemmatization
    DO_PUNCT_REMOVAL = "do_punct_removal"  # filter out tokens that are pure punctuations
    DO_LOWER = "do_lower"  # lowercase all the tokens
    # tokenizer
    TOKENIZER = "tokenizer"
    # reject threshold
    REJECT_THRESHOLD = "reject_threshold"

    def __init__(self, **kwargs):
        """
        initialization function for tokenization
        :param kwargs:
        """
        self.min_word_len = kwargs.pop(self.MIN_WORD_LEN, 1)
        self.max_word_len = kwargs.pop(self.MAX_WORD_LEN, 25)
        self.space_chars = kwargs.pop(self.SPACE_CHARS, [])
        self.do_stop_words = kwargs.pop(self.DO_STOP_WORDS, False)
        if self.do_stop_words:
            self.stop_words = set(kwargs.pop(self.STOP_WORDS,
                                             stopwords.words('english')))
        elif self.STOP_WORDS in kwargs:
            raise ValueError("please set do_stop_words to True when stop_words are provided.")

        self.passage_len = kwargs.pop(self.PASSAGE_LEN, 100)
        self.max_sent_len = kwargs.pop(self.MAX_SENT_LEN, 35)
        self.respect_sent_boundaries = kwargs.pop(self.RESPECT_SENT_BOUNDARIES, False)
        self.do_sliding_window_passages = kwargs.pop(self.DO_SLIDING_WINDOW_PASSAGES, False)
        self.do_char_offsets = kwargs.pop(self.DO_CHAR_OFFSETS, False)
        self.strip_inword_punctuation = kwargs.pop(self.STRIP_INWORD_PUNCTUATION, False)
        self.remove_numerics = kwargs.pop(self.REMOVE_NUMERICS, False)
        self.do_stem = kwargs.pop(self.DO_STEM, False)
        self.do_lemma = kwargs.pop(self.DO_LEMMA, False)
        self.do_punct_removal = kwargs.pop(self.DO_PUNCT_REMOVAL, False)
        self.do_lower = kwargs.pop(self.DO_LOWER, False)
        self.tokenizer = kwargs.pop(self.TOKENIZER, nltk.tokenize.word_tokenize)
        self.reject_threshold = kwargs.pop(self.REJECT_THRESHOLD, -1)

        if kwargs:
            raise ValueError("following keyword arguments not recognized:{}".format(kwargs))

        if self.do_stem and self.do_lemma:
            raise ValueError("stemming and lemmatization cannot be turned on simultaneously")

        self.stemmer = None
        if self.do_stem:
            self.stemmer = PorterStemmer()

        self.lemmatizer = None
        if self.do_lemma:
            self.lemmatizer = WordNetLemmatizer()

    def tokenize_passages(self, text):
        """
        function to split text into tokenized passages
        :param text:
        :return:list of list of list of TokenWithOffsets if <code>respect_sent_boundaries</code>
        is True
                    where outer list is passages, inner list is sentences, and innermost list is words with offsets
                 list of list of TokenWithOffsets if <code>respect_sent_boundaries</code> is False
                    where outerlist is passages, inner list is words with offsets
        """
        passages = []
        if self.respect_sent_boundaries:
            if self.do_sliding_window_passages:
                nonoverlapping_passages = self._tokenize_nonoverlapping_passages_sentence_boundaries(text)
                if len(nonoverlapping_passages) <= 1:
                    return nonoverlapping_passages
                overlapping_passages = []
                # get the half the sentences in left and right non-overlapping passages each into overlapping passages
                for i in range(len(nonoverlapping_passages) - 1):
                    left_passage_start_index = len(nonoverlapping_passages[i]) // 2
                    right_passage_end_index = max(1, len(nonoverlapping_passages[i + 1]) // 2)
                    overlapping_passages.append(nonoverlapping_passages[i][left_passage_start_index:]
                                                + nonoverlapping_passages[i + 1][:right_passage_end_index])

                # interleave the non-overlapping and overlapping passages
                for i in range(len(nonoverlapping_passages) - 1):
                    passages.append(nonoverlapping_passages[i])
                    passages.append(overlapping_passages[i])
                passages.append(nonoverlapping_passages[-1])
                return passages
            return self._tokenize_nonoverlapping_passages_sentence_boundaries(text)

        tokenized_words = self.tokenize(text)
        if self.do_sliding_window_passages:
            for i in range(0, len(tokenized_words), self.passage_len // 2):
                passages.append(tokenized_words[i:i + self.passage_len])
                if i + self.passage_len >= len(tokenized_words):
                    break
        else:
            passages = _break_tokenized_text_to_nonoverlapping_passages(tokenized_words, self.passage_len)
        return passages

    def tokenize_sentences(self, text):
        """
        function to tokenize a piece of text into sentences and then words within each sentence
        :param text: string
        :return: list of list of TokenWithOffsets
                 where outer list corresponds to sentences and inner list corresponds to words within each sentence
        """
        tokenized_sentences = []
        for sent_with_offsets in self._tokenize_text(text, sent_tok):
            sent_text = sent_with_offsets.get_token()
            sent_start = sent_with_offsets.get_start_offset()
            tokenized_sentences.append(self.tokenize(sent_text, sent_start))
        return tokenized_sentences

    def tokenize(self, text, offset=0):
        """
        function to tokenize a piece of text into a flat list of tokens
        :param text: string
        :param offset: char offset to include if any.
        :return: list of TokenWithOffsets if <code>do_char_offsets</code> is True
                 list of tokens otherwise
        """
        tokenized_words = self._tokenize_text(text, self.tokenizer, offset)
        transformed_words = self._filter_and_transform(tokenized_words)
        if not self.do_char_offsets:
            transformed_words = [token_with_offsets.get_token() for token_with_offsets in transformed_words]
        return transformed_words

    def _tokenize_nonoverlapping_passages_sentence_boundaries(self, text):
        """
        function to tokenize text into passages that respect sentence boundary, but without sliding window
        :param text: string
        :return: list of list of list of TokenWithOffsets where
                    outer list is for passages, inner list is for sentences and
                    inner most list is for words within each sentence.
        """
        passages = []
        tokenized_sentences = self.tokenize_sentences(text)
        tokenized_sentences = self._breakup_long_sentences(tokenized_sentences)
        sent_lens = [len(sent) for sent in tokenized_sentences]
        curr_passage = []
        curr_passage_len = 0
        for i, sent in enumerate(tokenized_sentences):
            curr_passage.append(sent)
            if i < len(tokenized_sentences) - 1 and sum(sent_lens[i + 1:]) <= self.passage_len / 2:
                # if the remaining sentences are less than half of passage length,
                # append them to the existing passage.
                curr_passage.extend(tokenized_sentences[i + 1:])
                passages.append(curr_passage)
                break
            curr_passage_len += len(sent)
            if curr_passage_len >= self.passage_len or i == len(tokenized_sentences) - 1:
                passages.append(curr_passage)
                curr_passage = []
                curr_passage_len = 0
        return passages

    def _breakup_long_sentences(self, tokenized_sentences):
        broken_sentences = []
        for tokenized_sent in tokenized_sentences:
            if len(tokenized_sent) > self.max_sent_len:
                sent_passages = _break_tokenized_text_to_nonoverlapping_passages(tokenized_sent, self.max_sent_len)
                broken_sentences.extend(sent_passages)
            else:
                broken_sentences.append(tokenized_sent)
        return broken_sentences

    def _tokenize_text(self, text, tokenizer, offset=0):
        """
        function to tokenize text into words or sentences
        :param text: string
        :param tokenizer: bert_tok, word_tok, or sent_tok
        :param offset: any global offset to be added to relative offsets.
        Useful when tokenizing text into sentences and words
        :return: list of (segment, (start_offset, end_offset)) tuples.
        """
        if not isinstance(text, str):
            raise ValueError('text type is invalid: {}'.format(type(text)))
        for space_char in self.space_chars:
            text = text.replace(space_char, ' ')
        start, end = 0, 0
        tokenized_segments = []
        text = self._remove_long_words(text, self.reject_threshold)
        segments = tokenizer(text)

        if tokenizer is nltk.word_tokenize:
            segments = self._split_long_words(segments)
        for segment in segments:
            segment_start = text.find(segment, start)
            if tokenizer is nltk.word_tokenize and segment in ["``", "''"]:
                # NLTK replaces double quotes with two opening or closing quotes in the tokenized text
                # NLTK also replaces triple quotes with two opening quotes and a single quote
                # See: https://stackoverflow.com/
                #          questions/32185072/nltk-word-tokenize-behaviour-for-double-quotation-marks-is-confusing
                # so we need special processing here
                # the original token in the text can be any one of these, let's try to match all of them
                # and find the first match among all, which will be the correct match

                quotes_match = QUOTES.search(text, start)

                if quotes_match:
                    segment = quotes_match.group(0)
                    segment_start = quotes_match.start()
                else:
                    segment_start = -1

            if segment_start < 0:
                raise ValueError("cannot find the segment %s in the text %s" % (segment, text))
            start = segment_start
            end = start + len(segment)
            tokenized_segments.append(TokenWithOffsets(segment, offset + start, offset + end))
            start = end
        return tokenized_segments

    def _split_long_words(self, segments):
        """
        splits long words that exceed MAX_WORD_LEN into shorter words
        :param segments: list of words
        :return: list of words that each of which doesn't exceed MAX_WORD_LEN
        """
        split_segments = []
        for seg in segments:
            if len(seg) <= self.max_word_len:
                split_segments.append(seg)
            else:
                for i in range(0, len(seg), self.max_word_len):
                    split_segments.append(seg[i:i+self.max_word_len])
        return split_segments

    def _remove_long_words(self, text, reject_threshold):
        """
        function only runs if reject_threshold is > 0. Reject all words longer than reject_threshold
        :param text: string of text
        :param reject_threshold: integer, reject words with len greather than this.
        :return: processed string with long words rejected.
        """
        if reject_threshold < 0:
            return text
        split_text = text.split(" ")
        checked_sentence = []
        skipped_count = 0
        for word in split_text:
            if len(word) < reject_threshold:
                checked_sentence.append(word)
            else:
                skipped_count += 1
                msg = 'skipping long word with length: {}'.format(len(word))
                logging.error(msg)
        msg = 'total skipped: {}, total kept: {}'.format(skipped_count, len(checked_sentence))
        logging.info(msg)
        return " ".join(checked_sentence)

    def _filter_and_transform(self, tokens_with_offsets):
        """
        function to apply transformations like stemming and lemmatization and filters such as stopword removal.
        :param tokens: list of (string, (start_offset, end_offset)) tuples
        :return: list of (string, (start_offset, end_offset)) tuples
        """
        transformed_tokens = []

        for token_w_offsets in tokens_with_offsets:
            if self.do_punct_removal and token_w_offsets.get_token() in self.PUNCTUATION_CHARS:
                continue
            if self.do_stop_words and token_w_offsets.get_token().lower() in self.stop_words:
                continue
            if self.do_lower:
                token_w_offsets.set_token(token_w_offsets.get_token().lower())
            if self.strip_inword_punctuation:
                token_w_offsets.set_token(token_w_offsets.get_token().translate(self.PUNCTUATION_TABLE))
            if len(token_w_offsets.get_token()) < self.min_word_len:
                continue
            if self.remove_numerics and token_w_offsets.get_token().isnumeric():
                continue

            if self.do_stem:
                token_w_offsets.set_token(self.stemmer.stem(token_w_offsets.get_token()))
            elif self.do_lemma:
                token_w_offsets.set_token(self.lemmatizer.lemmatize(token_w_offsets.get_token()))
            else:
                pass

            if token_w_offsets.get_token() is None:
                logging.error('tokenize produced None as a token')
                continue
            transformed_tokens.append(token_w_offsets)
        return transformed_tokens

    SETTINGS_BY_ALGO = {
        'DR': {
            MIN_WORD_LEN: 2,
            SPACE_CHARS: [
                "`",
                "'",
                "\"",
                "-",
                "/",
            ],
            DO_STOP_WORDS: True,
            STRIP_INWORD_PUNCTUATION: True,
            REMOVE_NUMERICS: True,
            DO_LEMMA: True,
            DO_PUNCT_REMOVAL: True,
            DO_LOWER: True,
        },
        'RC': {
            PASSAGE_LEN: 100,
            MAX_SENT_LEN: 35,
            RESPECT_SENT_BOUNDARIES: True,
            DO_PUNCT_REMOVAL: False,
            DO_CHAR_OFFSETS: True,
            DO_SLIDING_WINDOW_PASSAGES: True,
            DO_LOWER: True,
        },
        'FAQM': {},
    }

    @classmethod
    def for_algo(cls, algo_name):
        if algo_name not in cls.SETTINGS_BY_ALGO.keys():
            raise ValueError('algo_name not supported, expected {}: {}'.format(cls.SETTINGS_BY_ALGO.keys(), algo_name))
        return cls(**cls.SETTINGS_BY_ALGO[algo_name])


def _break_tokenized_text_to_nonoverlapping_passages(tokenized_words, passage_len):
    passages = []
    for i in range(0, len(tokenized_words), passage_len):
        if len(tokenized_words) - (i + passage_len) <= passage_len / 2:
            # look ahead and see if the remaining text is too short; if yes, append it to the current passage.
            passages.append(tokenized_words[i:])
            break
        passages.append(tokenized_words[i:i + passage_len])
    return passages
