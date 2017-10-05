/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.languagedetector.utils

import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.han.HierarchySequence
import com.kotlinnlp.simplednn.deeplearning.embeddings.EmbeddingsContainerByStrings

/**
 * The helper to tokenize texts.
 */
private val tokenizer = TextTokenizer()

/**
 * Convert the [CharSequence] into a [HierarchySequence] of Embeddings as input of the HAN.
 *
 * @param embeddings the Embeddings container from which to extract Embeddings vector
 * @param dropout the probability of dropout
 *
 * @return a HAN [HierarchySequence]
 */
fun CharSequence.toHierarchySequence(embeddings: EmbeddingsContainerByStrings, dropout: Double = 0.0) =
  HierarchySequence(*Array(
    size = this.length,
    init = { charIndex -> embeddings.getEmbedding(this[charIndex].toString(), dropout = dropout).array.values }
  ))

/**
 * Split the string in tokens by spacing and punctuation chars.
 *
 * @param maxTokensLength the max length of a token (longer tokens will be split)
 *
 * @return an ArrayList of tokens (strings)
 */
fun String.tokenize(maxTokensLength: Int): List<String> = tokenizer.tokenize(this, maxTokensLength = maxTokensLength)
