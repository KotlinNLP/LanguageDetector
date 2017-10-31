/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.languagedetector.utils

import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.han.HierarchySequence
import com.kotlinnlp.simplednn.deeplearning.embeddings.EmbeddingsMap

/**
 * Convert the [CharSequence] into a [HierarchySequence] of Embeddings as input of the HAN.
 *
 * @param embeddings the map of chars to embeddings vectors
 * @param dropout the probability of dropout
 *
 * @return a HAN [HierarchySequence]
 */
fun CharSequence.toHierarchySequence(embeddings: EmbeddingsMap<Char>, dropout: Double = 0.0) =
  HierarchySequence(*Array(
    size = this.length,
    init = { charIndex -> embeddings.get(this[charIndex], dropout = dropout).array.values }
  ))
