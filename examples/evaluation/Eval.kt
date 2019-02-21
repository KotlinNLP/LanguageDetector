/* Copyright 2016-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package evaluation

import com.kotlinnlp.languagedetector.LanguageDetector
import com.kotlinnlp.languagedetector.LanguageDetectorModel
import com.kotlinnlp.languagedetector.dataset.CorpusReader
import com.kotlinnlp.languagedetector.helpers.ValidationHelper
import com.kotlinnlp.languagedetector.utils.FrequencyDictionary
import com.kotlinnlp.languagedetector.utils.TextTokenizer
import com.kotlinnlp.neuraltokenizer.NeuralTokenizerModel
import java.io.File
import java.io.FileInputStream

/**
 * Execute an evaluation of a [LanguageDetector].
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) {

  val parsedArgs = CommandLineArguments(args)

  val model = parsedArgs.modelPath.let {
    println("Loading model from '$it'...")
    LanguageDetectorModel.load(FileInputStream(File(it)))
  }
  val cjkModel = parsedArgs.cjkTokenizerModelPath.let {
    println("Loading CJK NeuralTokenizer model from '$it'...")
    NeuralTokenizerModel.load(FileInputStream(File(it)))
  }
  val dictionary = parsedArgs.frequencyDictPath.let {
    println("Loading words frequency dictionary from '$it'...")
    FrequencyDictionary.load(FileInputStream(File(it)))
  }
  val testSet = parsedArgs.validationSetPath.let {
    println("Reading dataset from '$it'...")
    CorpusReader().read(file = File(it))
  }

  val textTokenizer = TextTokenizer(cjkModel)
  val langDetector = LanguageDetector(model = model, tokenizer = textTokenizer, frequencyDictionary = dictionary)
  val helper = ValidationHelper(langDetector)
  val accuracy: Double = helper.validate(testSet = testSet)

  println("\n-- START VALIDATION ON %d TEST SENTENCES".format(testSet.size))

  println()
  println("Accuracy: %.2f%%".format(100.0 * accuracy))

  println()
  println("Confusion matrix:")
  println()
  println(helper.confusionMatrix)
}
