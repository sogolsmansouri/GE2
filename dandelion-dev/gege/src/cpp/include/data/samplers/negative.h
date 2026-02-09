#pragma once

#include "storage/graph_storage.h"

std::tuple<torch::Tensor, torch::Tensor> batch_sample(torch::Tensor edges, int num_negatives, bool inverse = false);

torch::Tensor deg_negative_local_filter(torch::Tensor deg_sample_indices, torch::Tensor edges);

torch::Tensor compute_filter_corruption_cpu(shared_ptr<GegeGraph> graph, torch::Tensor edges, torch::Tensor corruption_nodes, bool inverse = false,
                                            bool global = false, LocalFilterMode local_filter_mode = LocalFilterMode::ALL,
                                            torch::Tensor deg_sample_indices = torch::Tensor());

torch::Tensor compute_filter_corruption_gpu(shared_ptr<GegeGraph> graph, torch::Tensor edges, torch::Tensor corruption_nodes, bool inverse = false,
                                            bool global = false, LocalFilterMode local_filter_mode = LocalFilterMode::ALL,
                                            torch::Tensor deg_sample_indices = torch::Tensor());

torch::Tensor compute_filter_corruption(shared_ptr<GegeGraph> graph, torch::Tensor edges, torch::Tensor corruption_nodes, bool inverse = false,
                                        bool global = false, LocalFilterMode local_filter_mode = LocalFilterMode::ALL,
                                        torch::Tensor deg_sample_indices = torch::Tensor());

torch::Tensor apply_score_filter(torch::Tensor scores, torch::Tensor filter);

/**
 * Samples the negative edges from a given batch.
 */
class NegativeSampler {
   public:
    virtual ~NegativeSampler() {}

    /**
     * Get negative edges from the given batch.
     * Return a tensor of node ids of shape [num_negs] or a [num_negs, 3] shaped tensor of negative edges.
     * @param inverse Sample for inverse edges
     * @return The negative nodes/edges sampled
     */
    virtual std::tuple<torch::Tensor, torch::Tensor> getNegatives(shared_ptr<GegeGraph> graph, torch::Tensor edges = torch::Tensor(),
                                                                  bool inverse = false) = 0;
    // serve as `select` function.

    virtual std::tuple<torch::Tensor, torch::Tensor> compute(torch::Tensor src_embeddings, torch::Tensor dst_embeddings, torch::Tensor dst_neg_embeddings, torch::Tensor src_neg_embeddings,
                                                             torch::Tensor src_embeddings_g, torch::Tensor dst_embeddings_g, torch::Tensor dst_neg_embeddings_g, torch::Tensor src_neg_embeddings_g,
                                                             int batch_num, int embedding_size, int chunk_num, int num_per_chunk, bool has_relations, bool use_inverse_relations) {
        throw GegeRuntimeException("NegativeSampling::compute needs override");
    }

    virtual std::tuple<torch::Tensor, torch::Tensor> sample(torch::Tensor dst_negs, torch::Tensor src_negs, torch::Tensor dst_negs_scores, torch::Tensor src_negs_scores,
                                                             int chunk_num, int num_per_chunk, int selected_negatives_num, bool has_relations, bool use_inverse_relations) {
        throw GegeRuntimeException("NegativeSampling::sample needs override");
    }
};

class CorruptNodeNegativeSampler : public NegativeSampler {
   public:
    int num_chunks_;
    int num_negatives_;
    float degree_fraction_;
    bool filtered_;
    LocalFilterMode local_filter_mode_;

    CorruptNodeNegativeSampler(int num_chunks, int num_negatives, float degree_fraction, bool filtered = false,
                               LocalFilterMode local_filter_mode = LocalFilterMode::DEG);

    std::tuple<torch::Tensor, torch::Tensor> getNegatives(shared_ptr<GegeGraph> graph, torch::Tensor edges = torch::Tensor(), bool inverse = false) override;
};

class CorruptRelNegativeSampler : public NegativeSampler {
   public:
    int num_chunks_;
    int num_negatives_;
    bool filtered_;

    CorruptRelNegativeSampler(int num_chunks, int num_negatives, bool filtered = false);

    std::tuple<torch::Tensor, torch::Tensor> getNegatives(shared_ptr<GegeGraph> graph, torch::Tensor edges = torch::Tensor(), bool inverse = false) override;
};

class NegativeEdgeSampler : public NegativeSampler {
   public:
    int num_chunks_;
    int num_negatives_;

    NegativeEdgeSampler(int num_chunks, int num_negatives, bool filtered = false);

    std::tuple<torch::Tensor, torch::Tensor> getNegatives(shared_ptr<GegeGraph> graph, torch::Tensor edges = torch::Tensor(), bool inverse = false) override;
};

// APIs.
class NegativeSamplingBase : public NegativeSampler {
   public:
    int num_chunks_;
    int num_negatives_;
    float degree_fraction_;
    bool filtered_;
    LocalFilterMode local_filter_mode_;

    NegativeSamplingBase(int num_chunks, int num_negatives, float degree_fraction, bool filtered = false, LocalFilterMode local_filter_mode = LocalFilterMode::DEG);

    std::tuple<torch::Tensor, torch::Tensor> getNegatives(shared_ptr<GegeGraph> graph, torch::Tensor edges = torch::Tensor(), bool inverse = false) override;
};

class RNS : public NegativeSamplingBase {
   public:
    RNS(int num_chunks, int num_negatives, float degree_fraction, bool filtered = false, LocalFilterMode local_filter_mode = LocalFilterMode::DEG)
       : NegativeSamplingBase(num_chunks, num_negatives, degree_fraction, filtered, local_filter_mode) {
           SPDLOG_INFO("NegativeSampling: Used RNS");
    }

    std::tuple<torch::Tensor, torch::Tensor> compute(torch::Tensor src_embeddings, torch::Tensor dst_embeddings, torch::Tensor dst_neg_embeddings, torch::Tensor src_neg_embeddings,
                                                             torch::Tensor src_embeddings_g, torch::Tensor dst_embeddings_g, torch::Tensor dst_neg_embeddings_g, torch::Tensor src_neg_embeddings_g,
                                                             int batch_num, int embedding_size, int chunk_num, int num_per_chunk, bool has_relations, bool use_inverse_relations) override {
        torch::Tensor dst_negs_scores;
        torch::Tensor src_negs_scores;
        return std::forward_as_tuple(dst_negs_scores, src_negs_scores);
    }

    std::tuple<torch::Tensor, torch::Tensor> sample(torch::Tensor dst_negs, torch::Tensor src_negs, torch::Tensor dst_negs_scores, torch::Tensor src_negs_scores,
                                                             int chunk_num, int num_per_chunk, int selected_negatives_num, bool has_relations, bool use_inverse_relations) override {
        return std::forward_as_tuple(dst_negs, src_negs);
    }

};

class DNS : public NegativeSamplingBase {
   public:
    DNS(int num_chunks, int num_negatives, float degree_fraction, bool filtered = false, LocalFilterMode local_filter_mode = LocalFilterMode::DEG)
       : NegativeSamplingBase(num_chunks, num_negatives, degree_fraction, filtered, local_filter_mode) {
           SPDLOG_INFO("NegativeSampling: Used DNS");
    }

    std::tuple<torch::Tensor, torch::Tensor> compute(torch::Tensor src_embeddings, torch::Tensor dst_embeddings, torch::Tensor dst_neg_embeddings, torch::Tensor src_neg_embeddings,
                                                             torch::Tensor src_embeddings_g, torch::Tensor dst_embeddings_g, torch::Tensor dst_neg_embeddings_g, torch::Tensor src_neg_embeddings_g,
                                                             int batch_num, int embedding_size, int chunk_num, int num_per_chunk, bool has_relations, bool use_inverse_relations) override {
        torch::Tensor dst_negs_scores;
        torch::Tensor src_negs_scores;
        // could have problem if batch_num < chunk_num when padding.
        torch::Tensor padded_src_embeddings = src_embeddings;
        torch::Tensor padded_dst_embeddings = dst_embeddings;
        if (num_per_chunk != batch_num / chunk_num) {
            int64_t new_size = num_per_chunk * chunk_num;
            torch::nn::functional::PadFuncOptions options({0, 0, 0, new_size - batch_num});
            padded_src_embeddings = torch::nn::functional::pad(src_embeddings, options);
            if (has_relations && use_inverse_relations) {
                padded_dst_embeddings = torch::nn::functional::pad(dst_embeddings, options);
            }
        }
        padded_src_embeddings = padded_src_embeddings.view({chunk_num, num_per_chunk, embedding_size});
        dst_negs_scores = padded_src_embeddings.bmm(dst_neg_embeddings.transpose(1, 2));
        if (has_relations && use_inverse_relations) {
            padded_dst_embeddings = padded_dst_embeddings.view({chunk_num, num_per_chunk, embedding_size});
            src_negs_scores = padded_dst_embeddings.bmm(src_neg_embeddings.transpose(1, 2));
        }
        return std::forward_as_tuple(dst_negs_scores, src_negs_scores);
    }

    std::tuple<torch::Tensor, torch::Tensor> sample(torch::Tensor dst_negs, torch::Tensor src_negs, torch::Tensor dst_negs_scores, torch::Tensor src_negs_scores,
                                                             int chunk_num, int num_per_chunk, int selected_negatives_num, bool has_relations, bool use_inverse_relations) override {
        torch::Tensor selected_dst_negs;
        torch::Tensor selected_src_negs;

        auto dst_results = dst_negs_scores.topk(selected_negatives_num, 2);
        torch::Tensor dst_results_scores = std::get<0>(dst_results);
        torch::Tensor dst_results_indices = std::get<1>(dst_results);  // chunk_num x num_per_chunk x  selected_negatives_num
        dst_results_indices = dst_results_indices.view({chunk_num, num_per_chunk * selected_negatives_num});
        selected_dst_negs = dst_negs.gather(1, dst_results_indices);

        if (has_relations && use_inverse_relations) {
            auto src_results = src_negs_scores.topk(selected_negatives_num, 2);
            torch::Tensor src_results_scores = std::get<0>(src_results);
            torch::Tensor src_results_indices = std::get<1>(src_results);
            src_results_indices = src_results_indices.view({chunk_num, num_per_chunk * selected_negatives_num});
            selected_src_negs = dst_negs.gather(1, src_results_indices);
        }

        return std::forward_as_tuple(selected_dst_negs, selected_src_negs);
    }
};

class KBGAN : public NegativeSamplingBase {
   public:
    KBGAN(int num_chunks, int num_negatives, float degree_fraction, bool filtered = false, LocalFilterMode local_filter_mode = LocalFilterMode::DEG)
       : NegativeSamplingBase(num_chunks, num_negatives, degree_fraction, filtered, local_filter_mode) {
           SPDLOG_INFO("NegativeSampling: Used KBGAN");
    }

    std::tuple<torch::Tensor, torch::Tensor> compute(torch::Tensor src_embeddings, torch::Tensor dst_embeddings, torch::Tensor dst_neg_embeddings, torch::Tensor src_neg_embeddings,
                                                             torch::Tensor src_embeddings_g, torch::Tensor dst_embeddings_g, torch::Tensor dst_neg_embeddings_g, torch::Tensor src_neg_embeddings_g,
                                                             int batch_num, int embedding_size, int chunk_num, int num_per_chunk, bool has_relations, bool use_inverse_relations) override {
        torch::Tensor dst_negs_scores;
        torch::Tensor src_negs_scores;
        // could have problem if batch_num < chunk_num when padding.
        torch::Tensor padded_src_embeddings_g = src_embeddings_g;
        torch::Tensor padded_dst_embeddings_g = dst_embeddings_g;
        if (num_per_chunk != batch_num / chunk_num) {
            int64_t new_size = num_per_chunk * chunk_num;
            torch::nn::functional::PadFuncOptions options({0, 0, 0, new_size - batch_num});
            padded_src_embeddings_g = torch::nn::functional::pad(src_embeddings_g, options);
            if (has_relations && use_inverse_relations) {
                padded_dst_embeddings_g = torch::nn::functional::pad(dst_embeddings_g, options);
            }
        }
        padded_src_embeddings_g = padded_src_embeddings_g.view({chunk_num, num_per_chunk, embedding_size});
        dst_negs_scores = padded_src_embeddings_g.bmm(dst_neg_embeddings_g.transpose(1, 2));
        if (has_relations && use_inverse_relations) {
            padded_dst_embeddings_g = padded_dst_embeddings_g.view({chunk_num, num_per_chunk, embedding_size});
            src_negs_scores = padded_dst_embeddings_g.bmm(src_neg_embeddings_g.transpose(1, 2));
        }
        return std::forward_as_tuple(dst_negs_scores, src_negs_scores);
    }

    std::tuple<torch::Tensor, torch::Tensor> sample(torch::Tensor dst_negs, torch::Tensor src_negs, torch::Tensor dst_negs_scores, torch::Tensor src_negs_scores,
                                                             int chunk_num, int num_per_chunk, int selected_negatives_num, bool has_relations, bool use_inverse_relations) override {
        torch::Tensor selected_dst_negs;
        torch::Tensor selected_src_negs;

        auto dst_results = dst_negs_scores.topk(selected_negatives_num, 2);
        torch::Tensor dst_results_scores = std::get<0>(dst_results);
        torch::Tensor dst_results_indices = std::get<1>(dst_results);  // chunk_num x num_per_chunk x  selected_negatives_num
        dst_results_indices = dst_results_indices.view({chunk_num, num_per_chunk * selected_negatives_num});
        selected_dst_negs = dst_negs.gather(1, dst_results_indices);

        if (has_relations && use_inverse_relations) {
            auto src_results = src_negs_scores.topk(selected_negatives_num, 2);
            torch::Tensor src_results_scores = std::get<0>(src_results);
            torch::Tensor src_results_indices = std::get<1>(src_results);
            src_results_indices = src_results_indices.view({chunk_num, num_per_chunk * selected_negatives_num});
            selected_src_negs = dst_negs.gather(1, src_results_indices);
        }

        return std::forward_as_tuple(selected_dst_negs, selected_src_negs);
    }
};
