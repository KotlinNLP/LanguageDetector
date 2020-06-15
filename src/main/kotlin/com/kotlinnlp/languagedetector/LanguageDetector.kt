/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.languagedetector

import com.kotlinnlp.languagedetector.utils.*
import com.kotlinnlp.linguisticdescription.language.Language
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.deeplearning.attention.han.HANEncoder
import com.kotlinnlp.simplednn.deeplearning.attention.han.HierarchySequence
import com.kotlinnlp.simplednn.simplemath.exp
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import java.util.*

/**
 * A language detector based on Hierarchic Attention Networks.
 * If the [frequencyDictionary] is not null it is used to boost the predictions.
 *
 * @property model the model of this [LanguageDetector]
 * @property tokenizer the [TextTokenizer] to tokenize input texts
 * @property frequencyDictionary the words frequency dictionary (default = null)
 */
class LanguageDetector(
  val model: LanguageDetectorModel,
  val tokenizer: TextTokenizer,
  val frequencyDictionary: FrequencyDictionary? = null
) {

  /**
   *
   */
  data class TokenClassification(val languages: DenseNDArray, val charsImportance: DenseNDArray)

  /**
   * The encoder of the input.
   */
  private val encoder = HANEncoder<DenseNDArray>(model = this.model.han, propagateToInput = true)

  /**
   * Detect the [Language] of the given [text].
   *
   * @param text the input text
   *
   * @return the detected [Language] for the given [text]
   */
  fun detectLanguage(text: String): Language = this.getLanguage(prediction = this.predict(text))

  /**
   * Get the predicted [Language] from the given [prediction].
   *
   * @param prediction a prediction array of this [LanguageDetector], as output of the [predict] method
   *
   * @return the predicted [Language]
   */
  fun getLanguage(prediction: DenseNDArray): Language =
    if (prediction.sum() == 0.0) Language.Unknown else this.model.supportedLanguages[prediction.argMaxIndex()]

  /**
   * Get the full distribution of languages within a prediction, sorted by descending score.
   *
   * @param prediction a prediction array of this [LanguageDetector], as output of the [predict] method
   *
   * @return the list of all the possible languages with the related prediction score, sorted by descending score
   */
  fun getFullDistribution(prediction: DenseNDArray): List<Pair<Language, Double>> =
    this.model.supportedLanguages
      .zip(prediction.toDoubleArray().toList())
      .sortedByDescending { it.second }

  /**
   * Get the languages of the given [text] as probability distribution.
   *
   * @param text the input text
   *
   * @return the probability distribution of the predicted languages as [DenseNDArray]
   */
  fun predict(text: String): DenseNDArray {

    val classifications = mutableListOf<DenseNDArray>()

    this.forEachToken(text) { token ->

      val tokenClassification = this.classifyToken(token)

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
   * @return the list of token classifications, as Pairs of <token, [TokenClassification]>
   */
  fun classifyTokens(text: String): List<Pair<String, TokenClassification>> {

    val tokensClassifications = mutableListOf<Pair<String, TokenClassification>>()

    this.forEachToken(text) { token ->

      var segmentClassification = this.classifyToken(token)

      if (this.frequencyDictionary != null) {
        val tokenFreq: DenseNDArray? = this.frequencyDictionary.getFreqOf(token)

        if (tokenFreq != null) {
          segmentClassification = this.combineClassifications(listOf(segmentClassification, tokenFreq))
        }
      }

      @Suppress("UNCHECKED_CAST")
      val classification = TokenClassification(
        languages = segmentClassification,
        charsImportance = (this.encoder.getInputImportanceScores() as HierarchySequence<DenseNDArray>)[0])

      tokensClassifications.add(Pair(token, classification))
    }

    return tokensClassifications.toList()
  }

  /**
   * Classify the language of a single token.
   *
   * @param token the word to classify (classes are all supported languages)
   * @param dropout the probability of dropout of the Embeddings
   *
   * @return a [DenseNDArray] containing the languages classification of the [token]
   */
  fun classifyToken(token: String, dropout: Double = 0.0): DenseNDArray {

    require(token.isNotEmpty()) { "Empty chars sequence" }

    return this.encoder.forward(token.toHierarchySequence(this.model.embeddings, dropout = dropout)).copy()
  }

  /**
   * Execute the backward of the internal HAN classifier using the given [outputErrors].
   *
   * @param outputErrors the errors of the output
   */
  fun backward(outputErrors: DenseNDArray) {
    this.encoder.backward(outputErrors = outputErrors)
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the HAN parameters
   */
  fun getParamsErrors(copy: Boolean = true): ParamsErrorsList = this.encoder.getParamsErrors(copy = copy)

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the input sequence of tokens
   */
  @Suppress("UNCHECKED_CAST")
  fun getInputSequenceErrors(copy: Boolean = true): ArrayList<DenseNDArray> =
    this.encoder.getInputErrors(copy = copy) as HierarchySequence<DenseNDArray>

  /**
   * Tokenize the given [text] and yield its tokens.
   *
   * @param text the input text
   * @param callback a callback called for each token, passing it as argument
   */
  internal fun forEachToken(text: String, callback: (String) -> Unit) {

    this@LanguageDetector.tokenizer.tokenize(text, maxTokensLength = this@LanguageDetector.model.maxTokensLength)
      .forEach { token -> callback(token) }
  }

  /**
   * Combine classifications of more tokens in a single one by making their logarithmic sum and returning a normalized
   * conditional probability.
   *
   * @param classifications an [ArrayList] of classifications (one for each text segment)
   *
   * @return a [DenseNDArray] containing the combined classification as conditional probability
   */
  private fun combineClassifications(classifications: List<DenseNDArray>): DenseNDArray {

    val logSum: DenseNDArray = classifications[0].ln()

    classifications.subList(1, classifications.size).forEach { logSum.assignSum(it.ln()) }

    val logSumNorm: DenseNDArray = logSum.assignSub(logSum.max())
    val linearSumNorm: DenseNDArray = exp(logSumNorm)

    return linearSumNorm.assignDiv(linearSumNorm.sum())
  }
}
