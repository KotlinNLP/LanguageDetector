/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.languagedetector.helpers

import com.kotlinnlp.languagedetector.LanguageDetector
import com.kotlinnlp.languagedetector.dataset.Example
import com.kotlinnlp.linguisticdescription.language.Language
import com.kotlinnlp.utils.ConfusionMatrix
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar

/**
 * A helper for the validation of a [LanguageDetector].
 */
class ValidationHelper(private val languageDetector: LanguageDetector) {

  /**
   * The confusion matrix of the predictions.
   */
  val confusionMatrix = ConfusionMatrix(this.languageDetector.model.supportedLanguages.map { it.isoCode })

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

    this.confusionMatrix.reset()

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
   * Validate the [languageDetector] with the given [example].
   *
   * @param example an example of the validation dataset
   *
   * @return 1 if the prediction is correct, 0 otherwise
   */
  private fun validateExample(example: Example): Int {

    val predictedLang: Language = this.languageDetector.detectLanguage(example.text)

    if (predictedLang != Language.Unknown)
      this.confusionMatrix.increment(expected = example.language.id, found = predictedLang.id)

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
