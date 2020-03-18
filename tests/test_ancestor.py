#! /usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from mutyper.ancestor import Ancestor


class TestAncestor(unittest.TestCase):
    def setUp(self):
        self.anc = Ancestor('tests/test_data/ancestor.fa')

    def test_seq(self):
        anc_seq = self.anc.fasta['foo'][:]
        anc_seq_true = 'AAACCCgggTTT'
        self.assertEqual(anc_seq, anc_seq_true)

    def test_context(self):
        self.assertEqual(self.anc.context('foo', 1), 'AAA')
        self.assertEqual(self.anc.context('foo', 2), 'AAC')
        self.assertEqual(self.anc.context('foo', 3), 'ACC')
        self.assertEqual(self.anc.context('foo', 4), 'CCC')
        self.assertEqual(self.anc.context('foo', 5), None)
        self.assertEqual(self.anc.context('foo', 6), None)
        self.assertEqual(self.anc.context('foo', 7), None)
        self.assertEqual(self.anc.context('foo', 8), None)
        self.assertEqual(self.anc.context('foo', 9), None)
        self.assertEqual(self.anc.context('foo', 10), 'AAA')

    def test_mutation_type(self):
        self.assertEqual(self.anc.mutation_type('foo', 1, 'A', 'T'),
                         ('AAA', 'ATA'))
        # infinite sites violation
        self.assertEqual(self.anc.mutation_type('foo', 1, 'C', 'T'),
                         (None, None))
        # low confidence (lower case) ancestral state
        self.assertEqual(self.anc.mutation_type('foo', 7, 'G', 'T'),
                         (None, None))
        # reverse complement
        self.assertEqual(self.anc.mutation_type('foo', 10, 'G', 'T'),
                         ('AAA', 'ACA'))


if __name__ == '__main__':
    unittest.main()
