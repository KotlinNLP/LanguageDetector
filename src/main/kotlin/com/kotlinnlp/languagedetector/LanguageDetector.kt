/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.languagedetector

import com.kotlinnlp.languagedetector.utils.FrequencyDictionary
import com.kotlinnlp.languagedetector.utils.Language
import com.kotlinnlp.languagedetector.utils.toHierarchySequence
import com.kotlinnlp.languagedetector.utils.tokenize
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.han.HANEncoder
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.han.HANParameters
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.han.HierarchySequence
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import java.util.*

/**
 * A language detector based on Hierarchic Attention Networks.
 * If the [frequencyDictionary] is not null it is used to boost the predictions.
 *
 * @property model the model of this [LanguageDetector]
 * @property frequencyDictionary the words frequency dictionary (default = null)
 */
class LanguageDetector(val model: LanguageDetectorModel, val frequencyDictionary: FrequencyDictionary? = null) {

  /**
   * The encoder of the input.
   */
  private val encoder = HANEncoder<DenseNDArray>(model = model.han)

  /**
   * Detect the [Language] of the given [text].
   *
   * @param text the input text
   *
   * @return the detected [Language] for the given [text]
   */
  fun detectLanguage(text: String): Language {

    val prediction: DenseNDArray = this.predict(text)

    return if (prediction.sum() == 0.0) {
      Language.Unknown
    } else {
      this.model.supportedLanguages[prediction.argMaxIndex()]
    }
  }

  /**
   * Get the languages of the given [text] as probability distribution.
   *
   * @param text the input text
   *
   * @return the probability distribution of the predicted languages as [DenseNDArray]
   */
  fun predict(text: String): DenseNDArray {

    val classifications = mutableListOf<DenseNDArray>()

    text.tokenize(maxTokensLength = this.model.maxTokensLength).forEach { token ->

      val tokenClassification = this.forward(token)

      classifications.add(tokenClassification)

      if (this.frequencyDictionary != null) {
        val tokenFreq: DenseNDArray? = this.frequencyDictionary.getFreqOf(token)

        if (tokenFreq != null) {
          classifications.add(tokenFreq)
        }
      }
    }

    return if (classifications.isNotEmpty()) {
      this.combineClassifications(classifications)
    } else {
      DenseNDArrayFactory.zeros(Shape(this.model.supportedLanguages.size))
    }
  }

  /**
   * Get the classification for each token of the given [text].
   *
   * @param text the input text
   *
   * @return a list of classifications (as [DenseNDArray]s) for each token
   */
  fun classifyTokens(text: String): List<DenseNDArray> {

    val tokensClassifications = mutableListOf<DenseNDArray>()

    text.tokenize(maxTokensLength = this.model.maxTokensLength).forEach { token ->

      var segmentClassification = this.forward(token)

      if (this.frequencyDictionary != null) {
        val tokenFreq: DenseNDArray? = this.frequencyDictionary.getFreqOf(token)

        if (tokenFreq != null) {
          segmentClassification = this.combineClassifications(listOf(segmentClassification, tokenFreq))
        }
      }

      tokensClassifications.add(segmentClassification)
    }

    return tokensClassifications.toList()
  }

  /**
   * Forward the internal classifier given a token.
   *
   * @param token the word to classify (classes are all supported languages)
   * @param dropout the probability of dropout of the Embeddings
   *
   * @return a [DenseNDArray] containing the languages classification of the [token]
   */
  fun forward(token: String, dropout: Double = 0.0): DenseNDArray {
    require(token.isNotEmpty()) { "Empty chars sequence" }

    return this.encoder.forward(token.toHierarchySequence(this.model.embeddings, dropout = dropout))
  }

  /**
   * Execute the backward of the internal HAN classifier using the given [outputErrors].
   *
   * @param outputErrors the errors of the output
   */
  fun backward(outputErrors: DenseNDArray) {
    this.encoder.backward(outputErrors = outputErrors, propagateToInput = true)
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the HAN parameters
   */
  fun getParamsErrors(copy: Boolean = true): HANParameters = this.encoder.getParamsErrors(copy = copy)

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the input sequence of tokens
   */
  fun getInputSequenceErrors(copy: Boolean = true): ArrayList<DenseNDArray> {
    @Suppress("UNCHECKED_CAST")
    return this.encoder.getInputSequenceErrors(copy = copy) as HierarchySequence<DenseNDArray>
  }

  /**
   * Combine classifications of more tokens in a single one by multiplying them element-wise and normalizing respect
   * of the sum, returning a conditional probability.
   *
   * @param classifications an [ArrayList] of classifications (one for each text segment)
   *
   * @return a [DenseNDArray] containing the combined classification as conditional probability
   */
  private fun combineClassifications(classifications: List<DenseNDArray>): DenseNDArray {

    val combinedClassification: DenseNDArray = classifications[0].copy()

    (1 until classifications.size).forEach { i -> combinedClassification.assignProd(classifications[i]) }

    return combinedClassification.assignDiv(combinedClassification.sum()) // normalization
  }
}
