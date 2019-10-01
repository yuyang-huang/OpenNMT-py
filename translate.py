#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from itertools import repeat
import json

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser


def main(opt):
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    translator = build_translator(opt, report_score=True)
    src_shards = split_corpus(opt.src, opt.shard_size)
    tgt_shards = split_corpus(opt.tgt, opt.shard_size) \
        if opt.tgt is not None else repeat(None)
    mention_shards = (split_corpus(opt.mention, opt.shard_size)
                      if opt.mention is not None else repeat(None))
    shard_pairs = zip(src_shards, tgt_shards, mention_shards)

    for i, (src_shard, tgt_shard, mention_shard) in enumerate(shard_pairs):
        logger.info("Translating shard %d." % i)
        scores, predictions = translator.translate(
            src=src_shard,
            tgt=tgt_shard,
            mention=mention_shard,
            src_dir=opt.src_dir,
            batch_size=opt.batch_size,
            batch_type=opt.batch_type,
            attn_debug=opt.attn_debug
        )

        if opt.json_output:
            with open(opt.json_output, 'a') as f:
                for nbest_score, nbest_prediction in zip(scores, predictions):
                    docs = [{'score': float(score), 'prediction': prediction}
                            for score, prediction in zip(nbest_score, nbest_prediction)]
                    print(json.dumps(docs), file=f)


def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    main(opt)
