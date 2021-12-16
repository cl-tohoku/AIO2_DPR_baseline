#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 FAISS-based index components for dense retriver
 - https://github.com/facebookresearch/faiss/wiki
"""

import os
import logging
import pickle
from typing import List, Tuple

import faiss
import numpy as np

logger = logging.getLogger()

Vector = np.array
DataID = object                      # data id to be registered with faiss indexer
Score = float                        # value of vector operation (inner product if IP)
DataIDs = List[DataID]               # len(DataIDs) == top-k if search top-k
Scores = List[Score]                 # len(Scores) == top-k if search top-k
Res4Query = Tuple[DataIDs, Scores]   # search results for a query


class DenseIndexer(object):

    def __init__(self, buffer_size: int = 50000):
        self.buffer_size = buffer_size
        self.index_id_to_db_id = []
        self.index = None

    def index_data(self, data: List[Tuple[DataID, Vector]]):
        """ register data into faiss index """
        raise NotImplementedError

    def search_knn(self, query_vectors: Vector, top_docs: int) -> List[Res4Query]:
        """ search index corresponding to the query """
        raise NotImplementedError

    def serialize(self, file: str):
        """ dump index in pickle format """
        logger.info('Serializing index to %s', file)

        if os.path.isdir(file):
            index_file = os.path.join(file, "index.dpr")
            meta_file = os.path.join(file, "index_meta.dpr")
        else:
            index_file = file + '.index.dpr'
            meta_file = file + '.index_meta.dpr'

        faiss.write_index(self.index, index_file)
        with open(meta_file, mode='wb') as f:
            pickle.dump(self.index_id_to_db_id, f)

    def deserialize_from(self, file: str):
        """ load index from pickle file """
        logger.info('Loading index from %s', file)

        if os.path.isdir(file):
            index_file = os.path.join(file, "index.dpr")
            meta_file = os.path.join(file, "index_meta.dpr")
        else:
            index_file = file + '.index.dpr'
            meta_file = file + '.index_meta.dpr'

        self.index = faiss.read_index(index_file)
        logger.info('Loaded index of type %s and size %d', type(self.index), self.index.ntotal)

        with open(meta_file, "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
        assert len(
            self.index_id_to_db_id) == self.index.ntotal, 'Deserialized index_id_to_db_id should match faiss index size'

    def _update_id_mapping(self, db_ids: List):
        self.index_id_to_db_id.extend(db_ids)


class DenseFlatIndexer(DenseIndexer):

    def __init__(self, vector_sz: int, buffer_size: int = 50000):
        super(DenseFlatIndexer, self).__init__(buffer_size=buffer_size)
        self.index = faiss.IndexFlatIP(vector_sz)

    def index_data(self, data: List[Tuple[DataID, Vector]]):
        n = len(data)
        # indexing in batches is beneficial for many faiss index types
        for i in range(0, n, self.buffer_size):
            db_ids = [t[0] for t in data[i:i + self.buffer_size]]
            vectors = [np.reshape(t[1], (1, -1)) for t in data[i:i + self.buffer_size]]
            vectors = np.concatenate(vectors, axis=0)
            self._update_id_mapping(db_ids)
            self.index.add(vectors)

        indexed_cnt = len(self.index_id_to_db_id)
        logger.info('Total data indexed %d', indexed_cnt)

    def search_knn(self, query_vectors: Vector, top_docs: int) -> List[Res4Query]:
        scores, indexes = self.index.search(query_vectors, top_docs)
        # convert to external ids
        db_ids = [[self.index_id_to_db_id[i] for i in query_top_idxs] for query_top_idxs in indexes]
        result = [(db_ids[i], scores[i]) for i in range(len(db_ids))]
        return result


class DenseHNSWFlatIndexer(DenseIndexer):
    """
     Efficient index for retrieval. Note: default settings are for hugh accuracy but also high RAM usage
    """

    def __init__(self, vector_sz: int, buffer_size: int = 50000, store_n: int = 512
                 , ef_search: int = 128, ef_construction: int = 200):
        super(DenseHNSWFlatIndexer, self).__init__(buffer_size=buffer_size)

        # IndexHNSWFlat supports L2 similarity only
        # so we have to apply DOT -> L2 similairy space conversion with the help of an extra dimension
        index = faiss.IndexHNSWFlat(vector_sz + 1, store_n)
        index.hnsw.efSearch = ef_search
        index.hnsw.efConstruction = ef_construction
        self.index = index
        self.phi = 0

    def index_data(self, data: List[Tuple[DataID, Vector]]):
        n = len(data)

        # max norm is required before putting all vectors in the index to convert inner product similarity to L2
        if self.phi > 0:
            raise RuntimeError('DPR HNSWF index needs to index all data at once,'
                               'results will be unpredictable otherwise.')
        phi = 0
        for i, item in enumerate(data):
            id, doc_vector = item
            norms = (doc_vector ** 2).sum()
            phi = max(phi, norms)
        logger.info('HNSWF DotProduct -> L2 space phi={}'.format(phi))
        self.phi = 0

        # indexing in batches is beneficial for many faiss index types
        for i in range(0, n, self.buffer_size):
            db_ids = [t[0] for t in data[i:i + self.buffer_size]]
            vectors = [np.reshape(t[1], (1, -1)) for t in data[i:i + self.buffer_size]]

            norms = [(doc_vector ** 2).sum() for doc_vector in vectors]
            aux_dims = [np.sqrt(phi - norm) for norm in norms]
            hnsw_vectors = [np.hstack((doc_vector, aux_dims[i].reshape(-1, 1))) for i, doc_vector in
                            enumerate(vectors)]
            hnsw_vectors = np.concatenate(hnsw_vectors, axis=0)

            self._update_id_mapping(db_ids)
            self.index.add(hnsw_vectors)
            logger.info('data indexed %d', len(self.index_id_to_db_id))

        indexed_cnt = len(self.index_id_to_db_id)
        logger.info('Total data indexed %d', indexed_cnt)

    def search_knn(self, query_vectors: Vector, top_docs: int) -> List[Res4Query]:

        aux_dim = np.zeros(len(query_vectors), dtype='float32')
        query_nhsw_vectors = np.hstack((query_vectors, aux_dim.reshape(-1, 1)))
        logger.info('query_hnsw_vectors %s', query_nhsw_vectors.shape)
        scores, indexes = self.index.search(query_nhsw_vectors, top_docs)
        # convert to external ids
        db_ids = [[self.index_id_to_db_id[i] for i in query_top_idxs] for query_top_idxs in indexes]
        result = [(db_ids[i], scores[i]) for i in range(len(db_ids))]
        return result

    def deserialize_from(self, file: str):
        super(DenseHNSWFlatIndexer, self).deserialize_from(file)
        # to trigger warning on subsequent indexing
        self.phi = 1
