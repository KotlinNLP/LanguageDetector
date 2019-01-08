/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.languagedetector.LanguageDetector
import com.kotlinnlp.languagedetector.LanguageDetectorModel
import com.kotlinnlp.languagedetector.utils.FrequencyDictionary
import com.kotlinnlp.linguisticdescription.language.Language
import com.kotlinnlp.languagedetector.utils.TextTokenizer
import com.kotlinnlp.neuraltokenizer.NeuralTokenizerModel
import java.io.File
import java.io.FileInputStream

/**
 * Detect the language of a text.
 *
 * Command line arguments:
 *  1. The filename of the [LanguageDetector] serialized model.
 *  2. The filename of the CJK NeuralTokenizer serialized model.
 *  3. The filename of the [FrequencyDictionary] serialized model.
 */
fun main(args: Array<String>) {

  val model = args[0].let {
    println("Loading model from '$it'... ")
    LanguageDetectorModel.load(FileInputStream(File(it)))
  }

  val cjkModel = args[1].let {
    println("Loading CJK NeuralTokenizer model from '$it'...")
    NeuralTokenizerModel.load(FileInputStream(File(it)))
  }

  val dictionary = args[2].let {
    println("Loading dictionary from '$it'... ")
    FrequencyDictionary.load(FileInputStream(File(it)))
  }

  val textTokenizer = TextTokenizer(cjkModel)
  val langDetector = LanguageDetector(model = model, tokenizer = textTokenizer, frequencyDictionary = dictionary)

  var text = readInput()

  while (text.isNotEmpty()) {

    val lang: Language = langDetector.detectLanguage(text)
    println("Detected language: ${lang.name}")

    text = readInput()
  }

  println("Thank you, bye!")
}

/**
 * Read a text from the standard input.
 *
 * @return the string read
 */
private fun readInput(): String {

  print("\nInsert a text (empty to exit): ")

  return readLine()!!.trim()
}
