/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.languagedetector.LanguageDetector
import com.kotlinnlp.languagedetector.LanguageDetectorModel
import com.kotlinnlp.languagedetector.dataset.CorpusReader
import com.kotlinnlp.languagedetector.helpers.ValidationHelper
import com.kotlinnlp.languagedetector.utils.FrequencyDictionary
import java.io.File
import java.io.FileInputStream

/**
 * Execute an evaluation of a [LanguageDetector], loading its serialized model from the file given as first argument.
 * The second argument is the filename of the [FrequencyDictionary] serialized model.
 * The third argument is the filename of the test set.
 */
fun main(args: Array<String>) {

  println("Loading model from '${args[0]}'...")
  val model = LanguageDetectorModel.load(FileInputStream(File(args[0])))

  println("Loading dictionary from '${args[1]}'...")
  val dictionary = FrequencyDictionary.load(FileInputStream(File(args[1])))

  println("Reading dataset from '${args[2]}'...")
  val testSet = CorpusReader().read(file = File(args[2]))

  println("\n-- START VALIDATION ON %d TEST SENTENCES".format(testSet.size))

  val helper = ValidationHelper(languageDetector = LanguageDetector(model = model, frequencyDictionary = dictionary))
  val accuracy: Double = helper.validate(testSet = testSet)

  println()
  println("Accuracy: %.2f%%".format(100.0 * accuracy))

  println()
  println("Confusion matrix:")
  println()
  println(helper.getFormattedConfusionMatrix())
}
