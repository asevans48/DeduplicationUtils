"""
A blocking record iterator using minLSH hash. Redis can be used to store
records for matching.

@author Andrew Evans
"""
import re

from datasketch import MinHash, MinHashLSH
from nltk.tokenize import word_tokenize
from sql.record.pgrecord_iterator import PGRecordIterator


class BlockingRecordIterator:
    """
    Blocking record iterator returning the record and matching row ids
    """

    def __init__(self,
                 id_name,
                 cursor_name,
                 conn,
                 query,
                 threshold,
                 storage_config,
                 is_letter=True,
                 is_text=False,
                 session_size=2000,
                 num_perm=128):
        """
        A blocking record iterator

        :param id_name: The id column name
        :param cursor_name: Name of the cursor for streaming
        :param conn:    The psycopg2 connection
        :param query:   Query to obtain the records
        :param threshold:   Jaccard similarity threshold for matching records
        :param storage_config:  Storage config for datasketch
        :param is_letter:   Whether to use letter shingles instead of words
        :param is_text: Whether this is a text
        :param seession_size:   Size of the session
        :param num_perm: Number of permutations
        """
        self.__id_name = id_name
        self.__cursor_name = cursor_name
        self.__conn = conn
        self.__query = query
        self.session_size = session_size
        self.__threshold = threshold
        self.__storage_config = storage_config
        self.__record_it = None
        self.__lsh = None
        self.__is_text = is_text
        self.__is_letter = is_letter
        self.__num_perm = num_perm
        self.__curr_it = None
        self.__record_it = None
        self.__hashes = []

    def close(self):
        """
        Close as necessary
        """
        if self.__record_it is not None:
            self.__record_it.close_cursor()
        if self.__conn:
            self.__conn.close()

    def __enter__(self):
        """
        Make closeable

        :return: Closeable version of self
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Close the connection
        """
        self.close()


    def _record_to_string(self, r, delimiter=""):
        """
        Converts a record to a string
        :param r:   The record to convert
        :param delimiter: join delimiter defaulting to no length string
        :return: A string representation of all items in teh record
        """
        vals = []
        keys = list(r.keys())
        if self.__id_name in keys:
            idx = keys.index(self.__id_name)
            keys.pop(idx)
        for k in keys:
            val = r[k]
            if val:
                vals.append(val)
        return delimiter.join([str(x) for x in vals])

    def _get_record_text(self, r):
        """
        Obtain the record text

        :param r:   The record
        :return:    The text from the non-id fields
        """
        txt = None
        rc = r.copy()
        keys = rc.keys()
        if self.__id_name in rc.keys():
            rc.pop(self.__id_name)
            keys = rc.keys()
        for key in keys:
            if txt is not None:
                txt = rc[key]
            else:
                tval = rc[key]
                txt = "{} {}".format(txt, tval)
        return txt

    def _split_str_to_chars(self, val):
        """
        Converts a string to a char set

        :param val: String to convert
        :return: A list of chars
        """
        if val:
            chrs = []
            for c in val:
                chrs.append(c)
            return chrs
        else:
            return []

    def _split_record_words(self, val):
        """
        Split a record to comparative words

        :param val: Split a record to words instead of letters
        :return:    The words
        """
        rstr = self._record_to_string(val, " ")
        return rstr.split(" ")

    def _create_word_shingles(self, val):
        """
        Create shingles with word tokenizer.

        :param val: The value to split
        :return: Obtain word shingles
        """
        return word_tokenize(val)

    def _get_shingle(self, r):
        """
        Obtain the shingle for lsh

        :param r:   The input row
        :return:    resulting set
        """
        if self.__is_letter:
            rstr = self._record_to_string(r)
            rstr = re.sub("\s+", "", rstr)
            cset = self._split_str_to_chars(rstr)
        elif self.__is_text:
            txt = self._get_record_text(r)
            cset = [
                x for x in txt.split(" ") if x is not None and len(x.trim()) > 0]
        else:
            cset = self._split_record_words(r)
        return cset

    def _get_min_hash(self, r):
        """

        :param r:   The incoming row
        :return:    resulting min hash
        """
        cset = self._get_shingle(r)
        m = MinHash(self.__num_perm)
        for c in cset:
            m.update(c.encode('utf-8'))
        return m

    def _create_hashes(self, it):
        """
        Create the hashes and insert into session.

        :param it:  The record iterator
        :return: Whether the end was reached or not
        """
        i = 0
        run = True
        with self.__lsh.insertion_session() as session:
            while i < self.session_size and run:
                try:
                    r = next(it)
                    keys = r.keys()
                    if self.__id_name in keys:
                        key_val = r[self.__id_name]
                        m = self._get_min_hash(r)
                        session.insert(key_val, m)
                    i += 1
                except StopIteration:
                    run = False
        return run

    def setup_lsh(self):
        """
        Create Minhash lsh
        """
        if self.__storage_config:
            self.__lsh = MinHashLSH(threshold=self.__threshold, num_perm=self.__num_perm, storage_config=self.__storage_config)
        else:
            raise ValueError("Storage Backend Required Due to Use of Session")

    def get_iter(self):
        """
        Obtain a record iterator and the iterator itself

        :return:    The iterator class and the iterator itself
        """
        it = PGRecordIterator(
            self.__conn, self.__query, itersize=self.session_size, name=self.__cursor_name)
        return (it, iter(it))

    def __iter__(self):
        """
        Create the iterator
        :return:
        """
        if self.__curr_it is None:
            self.setup_lsh()
            itcls, it = self.get_iter()
            self._create_hashes(it)
            it.close_cursor()
            cname = "{}{}".format(self.__cursor_name, "_matches")
            it.set_cursor_name(cname)
            it = iter(itcls)
            self.__curr_it = it
            self.__record_it = itcls
        return self

    def __next__(self):
        """
        Get the next record and related hash.
        :return: A tuple of the record and the related hash ids
        """
        try:
            nrow = next(self.__curr_it, None)
            if nrow:
                m = self._get_min_hash(nrow)
                vals = self.__lsh.query(m)
                return (nrow, vals)
            else:
                raise StopIteration
        except StopIteration:
            raise StopIteration
