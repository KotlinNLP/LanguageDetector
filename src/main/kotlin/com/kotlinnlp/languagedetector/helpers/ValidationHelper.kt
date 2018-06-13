/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.languagedetector.helpers

import com.kotlinnlp.languagedetector.LanguageDetector
import com.kotlinnlp.languagedetector.dataset.Example
import com.kotlinnlp.languagedetector.utils.Language
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar

/**
 * A helper for the validation of a [LanguageDetector].
 */
class ValidationHelper(private val languageDetector: LanguageDetector) {

  /**
   * The confusion matrix of the predictions.
   */
  private val confusionMatrix: DenseNDArray = DenseNDArrayFactory.zeros(
    Shape(this.languageDetector.model.supportedLanguages.size, this.languageDetector.model.supportedLanguages.size)
  )

  /**
   * When timing started.
   */
  private var startTime: Long = 0

  /**
   * Validate the [languageDetector] using the given test dataset and fill the confusion matrix properly.
   *
   * @param testSet the test dataset to validate the [languageDetector]
   * @param includeUnknown a Boolean indicating if the Unknown language must be considered in the counts
   *                       (default = false)
   *
   * @return the accuracy of the [languageDetector]
   */
  fun validate(testSet: ArrayList<Example>, includeUnknown: Boolean = false): Double {

    val progress = ProgressIndicatorBar(testSet.size)
    var correctPredictions = 0
    var validPredictions = 0

    this.startTiming()

    this.confusionMatrix.zeros()

    testSet.forEach { example ->

      progress.tick()

      if (example.language != Language.Unknown) {
        correctPredictions += this.validateExample(example = example)
      }

      if (example.language != Language.Unknown || includeUnknown) {
        validPredictions++
      }
    }

    println("Elapsed time: %s".format(this.formatElapsedTime()))

    return correctPredictions.toDouble() / validPredictions
  }

  /**
   * @return the confusion matrix of the last validation, formatted as [String]
   */
  fun getFormattedConfusionMatrix(): String {

    val languages = this.languageDetector.model.supportedLanguages

    var res = languages.joinToString(
      prefix = "    | ",
      transform = { lang -> "  %s  ".format(lang.isoCode) },
      separator = " | ",
      postfix = " \n")

    res += (0 until 9 * languages.size + 3).joinToString(transform = { "-" }, separator = "")
    res += "\n"

    res += (0 until this.confusionMatrix.shape.dim1).joinToString(
      transform = { i ->
        val row: DenseNDArray = this.confusionMatrix.getRow(i)
        val rowSum: Double = row.sum()
        val normRow: DenseNDArray = if (rowSum > 0.0) row.assignDiv(rowSum) else row

        (0 until this.confusionMatrix.shape.dim2).joinToString(
          prefix = " %s | ".format(languages[i].isoCode),
          transform = { j -> "%5.1f%%".format(100.0 * normRow[j]) },
          separator = " | ",
          postfix = " \n")
      },
      separator = "")

    return res
  }

  /**
   * Validate the [languageDetector] with the given [example].
   *
   * @param example an example of the validation dataset
   *
   * @return 1 if the prediction is correct, 0 otherwise
   */
  private fun validateExample(example: Example): Int {

    val predictedLang: Language = this.languageDetector.detectLanguage(example.text)

    if (predictedLang != Language.Unknown) {

      val row: Int = example.language.id
      val col: Int = predictedLang.id

      this.confusionMatrix[row, col] = this.confusionMatrix[row, col] + 1.0
    }

    return if (predictedLang == example.language) 1 else 0
  }

  /**
   * Start registering time.
   */
  private fun startTiming() {
    this.startTime = System.currentTimeMillis()
  }

  /**
   * @return the formatted string with elapsed time in seconds and minutes.
   */
  private fun formatElapsedTime(): String {

    val elapsedTime = System.currentTimeMillis() - this.startTime
    val elapsedSecs = elapsedTime / 1000.0

    return "%.3f s (%.1f min)".format(elapsedSecs, elapsedSecs / 60.0)
  }
}
