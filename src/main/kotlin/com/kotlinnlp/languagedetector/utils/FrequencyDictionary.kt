/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.languagedetector.utils

import com.kotlinnlp.linguisticdescription.language.Language
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.utils.Serializer
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable

/**
 * Map words with arrays of frequencies (one per language).
 */
class FrequencyDictionary: Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Read a [FrequencyDictionary] (serialized) from an input stream and decode it.
     *
     * @param inputStream the [InputStream] from which to read the serialized [FrequencyDictionary]
     *
     * @return the [FrequencyDictionary] read from [inputStream] and decoded
     */
    fun load(inputStream: InputStream): FrequencyDictionary = Serializer.deserialize(inputStream)
  }

  /**
   * A Boolean indicating if the frequencies have been normalized.
   */
  private var normalized: Boolean = false

  /**
   * The number of supported languages.
   */
  private val supportedLanguagesSize: Int = Language.values().size - 1

  /**
   * Map each word to its languages frequency.
   */
  private val freqMap = mutableMapOf<String, DenseNDArray>()

  /**
   * Word counts per language.
   */
  private val wordCountsPerLanguage: DenseNDArray = DenseNDArrayFactory.zeros(Shape(this.supportedLanguagesSize))

  /**
   * Serialize this [FrequencyDictionary] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [FrequencyDictionary]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)

  /**
   * Add the occurrence of the given [word] associated to the given [language].
   * The dictionary is required to be not normalized.
   *
   * @param word a word as String
   * @param language the [Language] associated to the [word]
   */
  fun addOccurrence(word: String, language: Language) {
    require(!this.normalized) { "Cannot add a word occurrence after normalization" }

    val lowerCaseWord = word.toLowerCase()

    if (lowerCaseWord !in freqMap) {
      freqMap[lowerCaseWord] = DenseNDArrayFactory.zeros(Shape(this.supportedLanguagesSize))
    }

    val wordFreq: DenseNDArray = freqMap[lowerCaseWord]!!

    wordFreq[language.id] = wordFreq[language.id] + 1
    this.wordCountsPerLanguage[language.id]++
  }

  /**
   * Normalize frequencies to probabilities and block dictionary.
   */
  fun normalize() {

    val eps: Double = 1.0 / this.freqMap.values.sumBy { it.sum().toInt() }

    this.freqMap.forEach { _, freqArray -> freqArray.normalizePerLanguage(zerosReplace = eps) }

    this.normalized = true
  }

  /**
   * @param word word as String
   *
   * @return the frequency [DenseNDArray] of the given [word]
   */
  fun getFreqOf(word: String): DenseNDArray? {
    return this.freqMap[word.toLowerCase()]
  }

  /**
   * Normalize an array of occurrences respect to the amount of words per language, removing zeros and normalizing it
   * to a probability.
   *
   * @param zerosReplace the value with which to replace zeros
   */
  private fun DenseNDArray.normalizePerLanguage(zerosReplace: Double) {

    // Normalize respect to the amount of words per language
    this.assignDiv(this@FrequencyDictionary.wordCountsPerLanguage)

    // Replace zeros
    (0 until this.length).forEach { langIndex -> if (this[langIndex] == 0.0) this[langIndex] = zerosReplace }

    // Normalize to probability
    this.assignDiv(this.sum())
  }
}
