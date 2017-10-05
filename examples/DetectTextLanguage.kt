/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.languagedetector.LanguageDetector
import com.kotlinnlp.languagedetector.LanguageDetectorModel
import com.kotlinnlp.languagedetector.utils.FrequencyDictionary
import com.kotlinnlp.languagedetector.utils.Language
import java.io.File
import java.io.FileInputStream

/**
 * Detect the language of a text.
 * The first argument is the filename of the [LanguageDetector] serialized model.
 * The second argument is the filename of the [FrequencyDictionary] serialized model.
 */
fun main(args: Array<String>) {

  println("Loading model from '${args[0]}'... ")
  val model = LanguageDetectorModel.load(FileInputStream(File(args[0])))

  println("Loading dictionary from '${args[1]}'... ")
  val dictionary = FrequencyDictionary.load(FileInputStream(File(args[1])))

  val langDetector = LanguageDetector(model = model, frequencyDictionary = dictionary)
  var exit = false

  while (!exit) {

    print("Insert text: ")
    val text = readLine()!!

    val lang: Language = langDetector.detectLanguage(text)
    println("Detected language: ${lang.name}")

    print("Do you want to continue? [Y|n] ")
    exit = readLine()!!.toLowerCase() == "n"
  }

  println("Thank you, bye!")
}
