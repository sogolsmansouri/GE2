#include "nn/decoders/edge/decoder_methods.h"

#include "configuration/options.h"

#include <iostream>

std::tuple<torch::Tensor, torch::Tensor> only_pos_forward(shared_ptr<EdgeDecoder> decoder, torch::Tensor edges, torch::Tensor node_embeddings) {
    torch::Tensor pos_scores;
    torch::Tensor inv_pos_scores;

    bool has_relations;
    if (edges.size(1) == 3) {
        has_relations = true;
    } else if (edges.size(1) == 2) {
        has_relations = false;
    } else {
        throw TensorSizeMismatchException(edges, "Edge list must be a 3 or 2 column tensor");
    }

    torch::Tensor src = node_embeddings.index_select(0, edges.select(1, 0));
    torch::Tensor dst = node_embeddings.index_select(0, edges.select(1, -1));

    torch::Tensor rel_ids;

    if (has_relations) {
        rel_ids = edges.select(1, 1);

        torch::Tensor rels = decoder->select_relations(rel_ids);

        pos_scores = decoder->compute_scores(decoder->apply_relation(src, rels), dst);

        if (decoder->use_inverse_relations_) {
            torch::Tensor inv_rels = decoder->select_relations(rel_ids, true);

            inv_pos_scores = decoder->compute_scores(decoder->apply_relation(dst, inv_rels), src);
        }
    } else {
        pos_scores = decoder->compute_scores(src, dst);
    }

    return std::forward_as_tuple(pos_scores, inv_pos_scores);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> neg_and_pos_forward(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges,
                                                                                           torch::Tensor negative_edges, torch::Tensor node_embeddings) {
    torch::Tensor pos_scores;
    torch::Tensor inv_pos_scores;
    torch::Tensor neg_scores;
    torch::Tensor inv_neg_scores;

    std::tie(pos_scores, inv_pos_scores) = only_pos_forward(decoder, positive_edges, node_embeddings);
    std::tie(neg_scores, inv_neg_scores) = only_pos_forward(decoder, negative_edges, node_embeddings);

    return std::forward_as_tuple(pos_scores, neg_scores, inv_pos_scores, inv_neg_scores);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> node_corrupt_forward(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges,
                                                                                            torch::Tensor node_embeddings, torch::Tensor dst_negs,
                                                                                            torch::Tensor src_negs) {
    torch::Tensor pos_scores;
    torch::Tensor inv_pos_scores;
    torch::Tensor neg_scores;
    torch::Tensor inv_neg_scores;
    bool has_relations;
    if (positive_edges.size(1) == 3) {
        has_relations = true;
    } else if (positive_edges.size(1) == 2) {
        has_relations = false;
    } else {
        throw TensorSizeMismatchException(positive_edges, "Edge list must be a 3 or 2 column tensor");
    }

    torch::Tensor src = node_embeddings.index_select(0, positive_edges.select(1, 0));
    torch::Tensor dst = node_embeddings.index_select(0, positive_edges.select(1, -1));
    torch::Tensor rel_ids;

    torch::Tensor dst_neg_embs = node_embeddings.index_select(0, dst_negs.flatten(0, 1)).reshape({dst_negs.size(0), dst_negs.size(1), -1});

    if (has_relations) {
        rel_ids = positive_edges.select(1, 1);
        torch::Tensor rels = decoder->select_relations(rel_ids);
        torch::Tensor adjusted_src = decoder->apply_relation(src, rels);
        pos_scores = decoder->compute_scores(adjusted_src, dst);
        neg_scores = decoder->compute_scores(adjusted_src, dst_neg_embs);

        if (decoder->use_inverse_relations_) {
            torch::Tensor inv_rels = decoder->select_relations(rel_ids, true);
            torch::Tensor adjusted_dst = decoder->apply_relation(dst, inv_rels);

            torch::Tensor src_neg_embs = node_embeddings.index_select(0, src_negs.flatten(0, 1)).reshape({src_negs.size(0), src_negs.size(1), -1});

            inv_pos_scores = decoder->compute_scores(adjusted_dst, src);
            inv_neg_scores = decoder->compute_scores(adjusted_dst, src_neg_embs);
        }
    } else {
        pos_scores = decoder->compute_scores(src, dst);
        neg_scores = decoder->compute_scores(src, dst_neg_embs);
    }

    if (pos_scores.size(0) != neg_scores.size(0)) {
        int64_t new_size = neg_scores.size(0) - pos_scores.size(0);
        torch::nn::functional::PadFuncOptions options({0, new_size});
        pos_scores = torch::nn::functional::pad(pos_scores, options);

        if (inv_pos_scores.defined()) {
            inv_pos_scores = torch::nn::functional::pad(inv_pos_scores, options);
        }
    }

    return std::forward_as_tuple(pos_scores, neg_scores, inv_pos_scores, inv_neg_scores);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rel_corrupt_forward(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges,
                                                                                           torch::Tensor node_embeddings, torch::Tensor neg_rel_ids) {
    torch::Tensor pos_scores;
    torch::Tensor inv_pos_scores;
    torch::Tensor neg_scores;
    torch::Tensor inv_neg_scores;

    if (positive_edges.size(1) != 3) {
        throw TensorSizeMismatchException(positive_edges, "Edge list must be a 3 column tensor");
    }

    torch::Tensor src = node_embeddings.index_select(0, positive_edges.select(1, 0));
    torch::Tensor dst = node_embeddings.index_select(0, positive_edges.select(1, -1));

    torch::Tensor rel_ids = positive_edges.select(1, 1);

    torch::Tensor rels = decoder->select_relations(rel_ids);
    torch::Tensor neg_rels = decoder->select_relations(neg_rel_ids);

    pos_scores = decoder->compute_scores(decoder->apply_relation(src, rels), dst);
    neg_scores = decoder->compute_scores(decoder->apply_relation(src, neg_rels), dst);

    if (decoder->use_inverse_relations_) {
        torch::Tensor inv_rels = decoder->select_relations(rel_ids, true);
        torch::Tensor inv_neg_rels = decoder->select_relations(neg_rel_ids, true);

        inv_pos_scores = decoder->compute_scores(decoder->apply_relation(dst, inv_rels), src);
        inv_neg_scores = decoder->compute_scores(decoder->apply_relation(dst, inv_neg_rels), src);
    }

    return std::forward_as_tuple(pos_scores, neg_scores, inv_pos_scores, inv_neg_scores);
}

std::tuple<torch::Tensor, torch::Tensor> prepare_pos_embeddings(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges, torch::Tensor src_embeddings, torch::Tensor dst_embeddings, bool has_relations) {
    torch::Tensor adjusted_src_embeddings;
    torch::Tensor adjusted_dst_embeddings;

    torch::Tensor rel_ids;
    torch::Tensor rels;
    torch::Tensor inv_rels;

    if (has_relations) {
        rel_ids = positive_edges.select(1, 1);
        rels = decoder->select_relations(rel_ids);
        adjusted_src_embeddings = decoder->apply_relation(src_embeddings, rels);
        if (decoder->use_inverse_relations_) {
            inv_rels = decoder->select_relations(rel_ids, true);
            adjusted_dst_embeddings = decoder->apply_relation(dst_embeddings, inv_rels);
        }
    } else {  // no relations
        adjusted_src_embeddings = src_embeddings;
    }

    return std::forward_as_tuple(adjusted_src_embeddings, adjusted_dst_embeddings);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> mod_node_corrupt_forward(NegativeSamplingMethod negative_sampling_method, float negative_sampling_selected_ratio, shared_ptr<NegativeSampler> negative_sampler,
                                                                                                shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges, torch::Tensor node_embeddings, torch::Tensor dst_negs, torch::Tensor src_negs,
                                                                                                torch::Tensor node_embeddings_g) {
    bool has_relations;
    if (positive_edges.size(1) == 3) {
        has_relations = true;
    } else if (positive_edges.size(1) == 2) {
        has_relations = false;
    } else {
        throw TensorSizeMismatchException(positive_edges, "Edge list must be a 3 or 2 column tensor");
    }

    torch::Tensor src_embeddings = node_embeddings.index_select(0, positive_edges.select(1, 0));
    torch::Tensor dst_embeddings = node_embeddings.index_select(0, positive_edges.select(1, -1));

    int batch_num = src_embeddings.sizes()[0];
    int embedding_size = src_embeddings.sizes()[1];
    int chunk_num = dst_negs.sizes()[0];
    int negatives_num = dst_negs.sizes()[1];
    int selected_negatives_num = int(negatives_num * negative_sampling_selected_ratio);
    int num_per_chunk = (int)ceil((float) batch_num / chunk_num);

    // SPDLOG_INFO("batch_num : {}", batch_num);
    // SPDLOG_INFO("embedding_size: {}", embedding_size);
    // SPDLOG_INFO("chunk_num: {}", chunk_num);
    // SPDLOG_INFO("negatives_num: {}", negatives_num);
    // SPDLOG_INFO("selected_negatives_num: {}", selected_negatives_num);

    torch::Tensor adjusted_src_embeddings;
    torch::Tensor adjusted_dst_embeddings;
    torch::Tensor dst_neg_embeddings;
    torch::Tensor src_neg_embeddings;

    auto all_pos_embeddings = prepare_pos_embeddings(decoder, positive_edges, src_embeddings, dst_embeddings, has_relations);

    if (has_relations) {
        adjusted_src_embeddings = std::get<0>(all_pos_embeddings);
        dst_neg_embeddings = node_embeddings.index_select(0, dst_negs.flatten(0, 1)).reshape({dst_negs.size(0), dst_negs.size(1), -1});
        if (decoder->use_inverse_relations_) {
            adjusted_dst_embeddings = std::get<1>(all_pos_embeddings);
            src_neg_embeddings = node_embeddings.index_select(0, src_negs.flatten(0, 1)).reshape({src_negs.size(0), src_negs.size(1), -1});
        }
    } else {
        adjusted_src_embeddings = std::get<0>(all_pos_embeddings);
        dst_neg_embeddings = node_embeddings.index_select(0, dst_negs.flatten(0, 1)).reshape({dst_negs.size(0), dst_negs.size(1), -1});
    }

    {
        torch::NoGradGuard no_grad;

        torch::Tensor src_embeddings_g;
        torch::Tensor dst_embeddings_g;
        torch::Tensor dst_neg_embeddings_g;
        torch::Tensor src_neg_embeddings_g;

        if (negative_sampling_method == NegativeSamplingMethod::GAN) {
            if (has_relations) {
                src_embeddings_g = node_embeddings_g.index_select(0, positive_edges.select(1, 0));
                dst_neg_embeddings_g = node_embeddings_g.index_select(0, dst_negs.flatten(0, 1)).reshape({dst_negs.size(0), dst_negs.size(1), -1});
                if (decoder->use_inverse_relations_) {
                    dst_embeddings_g = node_embeddings_g.index_select(0, positive_edges.select(1, -1));
                    src_neg_embeddings_g = node_embeddings_g.index_select(0, src_negs.flatten(0, 1)).reshape({src_negs.size(0), src_negs.size(1), -1});
                }
            } else {
                src_embeddings_g = node_embeddings_g.index_select(0, positive_edges.select(1, 0));
                dst_neg_embeddings_g = node_embeddings_g.index_select(0, dst_negs.flatten(0, 1)).reshape({dst_negs.size(0), dst_negs.size(1), -1});
            }
        }

        auto all_negs_scores = negative_sampler->compute(adjusted_src_embeddings, adjusted_dst_embeddings, dst_neg_embeddings, src_neg_embeddings,
                                              src_embeddings_g, dst_embeddings_g, dst_neg_embeddings_g, src_neg_embeddings_g,
                                              batch_num, embedding_size, chunk_num, num_per_chunk, has_relations, decoder->use_inverse_relations_);

        torch::Tensor dst_negs_scores = std::get<0>(all_negs_scores);
        torch::Tensor src_negs_scores = std::get<1>(all_negs_scores);

        auto all_selected_negs = negative_sampler->sample(dst_negs, src_negs, dst_negs_scores, src_negs_scores, chunk_num, num_per_chunk, selected_negatives_num,
                                                          has_relations, decoder->use_inverse_relations_);

        dst_negs = std::get<0>(all_selected_negs);
        src_negs = std::get<1>(all_selected_negs);
    }

    torch::Tensor pos_scores;
    torch::Tensor inv_pos_scores;
    torch::Tensor neg_scores;
    torch::Tensor inv_neg_scores;

    switch (negative_sampling_method) {
        case NegativeSamplingMethod::RNS : {
            torch::Tensor dst_neg_embs = node_embeddings.index_select(0, dst_negs.flatten(0, 1)).reshape({dst_negs.size(0), dst_negs.size(1), -1});
            if (has_relations) {
                pos_scores = decoder->compute_scores(adjusted_src_embeddings, dst_embeddings);
                neg_scores = decoder->compute_scores(adjusted_src_embeddings, dst_neg_embs);
                if (decoder->use_inverse_relations_) {
                    torch::Tensor src_neg_embs = node_embeddings.index_select(0, src_negs.flatten(0, 1)).reshape({src_negs.size(0), src_negs.size(1), -1});

                    inv_pos_scores = decoder->compute_scores(adjusted_dst_embeddings, src_embeddings);
                    inv_neg_scores = decoder->compute_scores(adjusted_dst_embeddings, src_neg_embs);
                }
            } else {
                pos_scores = decoder->compute_scores(adjusted_src_embeddings, dst_embeddings);
                neg_scores = decoder->compute_scores(adjusted_src_embeddings, dst_neg_embs);
            }
            break;
        }
        case NegativeSamplingMethod::DNS :
        case NegativeSamplingMethod::GAN : {
            torch::Tensor selected_dst_negs_embeddings = node_embeddings.index_select(0, dst_negs.flatten(0, 1)).reshape({chunk_num, num_per_chunk, selected_negatives_num, -1});
            // SPDLOG_INFO("selected_dst_negs_embeddings dim: {}", selected_dst_negs_embeddings.dim());
            // SPDLOG_INFO("selected_dst_negs_embeddings size[0]: {}", selected_dst_negs_embeddings.sizes()[0]);  // chunk num
            // SPDLOG_INFO("selected_dst_negs_embeddings size[1]: {}", selected_dst_negs_embeddings.sizes()[1]);  // num_per_chunk
            // SPDLOG_INFO("selected_dst_negs_embeddings size[2]: {}", selected_dst_negs_embeddings.sizes()[2]);  // selected negatives num
            // SPDLOG_INFO("selected_dst_negs_embeddings size[3]: {}", selected_dst_negs_embeddings.sizes()[3]);  // embedding size

            if (has_relations) {
                pos_scores = decoder->compute_scores(adjusted_src_embeddings, dst_embeddings);
                neg_scores = decoder->compute_scores(adjusted_src_embeddings, selected_dst_negs_embeddings);
                if (decoder->use_inverse_relations_) {
                    torch::Tensor selected_src_negs_embeddings = node_embeddings.index_select(0, src_negs.flatten(0, 1)).reshape({chunk_num, num_per_chunk, selected_negatives_num, -1});
                    inv_pos_scores = decoder->compute_scores(adjusted_dst_embeddings, src_embeddings);
                    inv_neg_scores = decoder->compute_scores(adjusted_dst_embeddings, selected_src_negs_embeddings);
                }
            } else {
                pos_scores = decoder->compute_scores(adjusted_src_embeddings, dst_embeddings);
                neg_scores = decoder->compute_scores(adjusted_src_embeddings, selected_dst_negs_embeddings);
            }
            break;
        }
        default : {
            throw GegeRuntimeException("Unsupported negative_sampling_method in scoreNegatives");
        }
    }

    if (pos_scores.size(0) != neg_scores.size(0)) {
        int64_t new_size = neg_scores.size(0) - pos_scores.size(0);
        torch::nn::functional::PadFuncOptions options({0, new_size});
        pos_scores = torch::nn::functional::pad(pos_scores, options);

        if (inv_pos_scores.defined()) {
            inv_pos_scores = torch::nn::functional::pad(inv_pos_scores, options);
        }
    }
    // SPDLOG_INFO("pos_scores : {}", pos_scores.dim());
    // SPDLOG_INFO("pos_scores size[0]: {}", pos_scores.sizes()[0]);  // chunk num
    // SPDLOG_INFO("pos_scores size[1]: {}", pos_scores.sizes()[1]);  // num_per_chunk
    // SPDLOG_INFO("pos_scores size[2]: {}", pos_scores.sizes()[2]);  // embedding size
    // SPDLOG_INFO("neg_scores: {}", neg_scores.dim());
    // SPDLOG_INFO("neg_scores size[0]: {}", neg_scores.sizes()[0]);  // chunk num
    // SPDLOG_INFO("neg_scores size[1]: {}", neg_scores.sizes()[1]);  // num_per_chunk
    // SPDLOG_INFO("neg_scores size[2]: {}", neg_scores.sizes()[2]);  // embedding size

    return std::forward_as_tuple(pos_scores, neg_scores, inv_pos_scores, inv_neg_scores);
}

std::tuple<torch::Tensor, torch::Tensor> get_rewards(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges, torch::Tensor node_embeddings, torch::Tensor dst_negs, torch::Tensor src_negs) {
    bool has_relations;
    if (positive_edges.size(1) == 3) {
        has_relations = true;
    } else if (positive_edges.size(1) == 2) {
        has_relations = false;
    } else {
        throw TensorSizeMismatchException(positive_edges, "Edge list must be a 3 or 2 column tensor");
    }
    torch::Tensor reward;
    torch::Tensor inv_reward;
    {
        torch::NoGradGuard no_grad;

        torch::Tensor adjusted_src_embeddings;
        torch::Tensor adjusted_dst_embeddings;
        torch::Tensor rel_ids;
        torch::Tensor rels;
        torch::Tensor inv_rels;

        torch::Tensor src_embeddings = node_embeddings.index_select(0, positive_edges.select(1, 0));
        torch::Tensor dst_embeddings = node_embeddings.index_select(0, positive_edges.select(1, -1));

        torch::Tensor dst_neg_embs = node_embeddings.index_select(0, dst_negs.flatten(0, 1)).reshape({dst_negs.size(0), dst_negs.size(1), -1});
        if (has_relations) {
            rel_ids = positive_edges.select(1, 1);
            rels = decoder->select_relations(rel_ids);
            adjusted_src_embeddings = decoder->apply_relation(src_embeddings, rels);
            // reward = decoder->compute_scores(adjusted_src_embeddings, dst_neg_embs);
            torch::Tensor logits = decoder->compute_scores(adjusted_src_embeddings, dst_neg_embs);
            reward = logits.sigmoid().sub(0.5).mul(2);
            if (decoder->use_inverse_relations_) {
                inv_rels = decoder->select_relations(rel_ids, true);
                adjusted_dst_embeddings = decoder->apply_relation(dst_embeddings, inv_rels);
                torch::Tensor src_neg_embs = node_embeddings.index_select(0, src_negs.flatten(0, 1)).reshape({src_negs.size(0), src_negs.size(1), -1});
                // inv_reward = decoder->compute_scores(adjusted_dst_embeddings, src_neg_embs);
                torch::Tensor inv_logits = decoder->compute_scores(adjusted_dst_embeddings, src_neg_embs);
                inv_reward = inv_logits.sigmoid().sub(0.5).mul(2);
            }
        } else {
            adjusted_src_embeddings = src_embeddings;
            // reward = decoder->compute_scores(adjusted_src_embeddings, dst_neg_embs);
            torch::Tensor logits = decoder->compute_scores(adjusted_src_embeddings, dst_neg_embs);
            reward = logits.sigmoid().sub(0.5).mul(2);
        }
    }
    return std::forward_as_tuple(reward, inv_reward);
}

torch::Tensor forward_g(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges, torch::Tensor node_embeddings_g, torch::Tensor dst_negs, torch::Tensor src_negs, torch::Tensor reward, torch::Tensor inv_reward) {
    bool has_relations;
    if (positive_edges.size(1) == 3) {
        has_relations = true;
    } else if (positive_edges.size(1) == 2) {
        has_relations = false;
    } else {
        throw TensorSizeMismatchException(positive_edges, "Edge list must be a 3 or 2 column tensor");
    }

    torch::Tensor src_embeddings_g = node_embeddings_g.index_select(0, positive_edges.select(1, 0));
    torch::Tensor dst_neg_embs_g = node_embeddings_g.index_select(0, dst_negs.flatten(0, 1)).reshape({dst_negs.size(0), dst_negs.size(1), -1});

    torch::Tensor logits = decoder->compute_scores(src_embeddings_g, dst_neg_embs_g);
    torch::Tensor probs = logits.softmax(1);
    reward = reward.mul(probs);
    torch::Tensor log_probs = logits.log_softmax(1);
    torch::Tensor loss_g = -(log_probs.mul(reward).mean());
    // torch::Tensor loss_g = -(log_probs.mul(reward).sum());

    if (has_relations && decoder->use_inverse_relations_) {
        torch::Tensor dst_embeddings_g = node_embeddings_g.index_select(0, positive_edges.select(1, -1));
        torch::Tensor src_neg_embs_g = node_embeddings_g.index_select(0, src_negs.flatten(0, 1)).reshape({src_negs.size(0), src_negs.size(1), -1});

        torch::Tensor inv_logits = decoder->compute_scores(dst_embeddings_g, src_neg_embs_g);
        torch::Tensor inv_probs = inv_logits.softmax(1);
        inv_reward = inv_reward.mul(inv_probs);
        torch::Tensor inv_log_probs = inv_logits.log_softmax(1);
        torch::Tensor inv_loss_g = -(inv_log_probs.mul(inv_reward).mean());
        // torch::Tensor inv_loss_g = -(inv_log_probs.mul(inv_reward).sum());

        return loss_g + inv_loss_g;
    }

    return loss_g;
}
