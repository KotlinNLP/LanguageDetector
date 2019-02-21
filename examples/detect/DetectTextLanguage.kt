/* Copyright 2016-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package detect

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
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) {

  val parsedArgs = CommandLineArguments(args)

  val model = parsedArgs.langDetectorModelPath.let {
    println("Loading model from '$it'... ")
    LanguageDetectorModel.load(FileInputStream(File(it)))
  }
  val cjkModel = parsedArgs.cjkTokenizerModelPath.let {
    println("Loading CJK NeuralTokenizer model from '$it'...")
    NeuralTokenizerModel.load(FileInputStream(File(it)))
  }
  val dictionary = parsedArgs.frequencyDictPath.let {
    println("Loading words frequency dictionary from '$it'... ")
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
