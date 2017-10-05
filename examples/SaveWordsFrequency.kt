/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.languagedetector.dataset.CorpusReader
import com.kotlinnlp.languagedetector.dataset.Example
import com.kotlinnlp.languagedetector.utils.FrequencyDictionary
import com.kotlinnlp.languagedetector.utils.tokenize
import com.kotlinnlp.simplednn.utils.progressindicator.ProgressIndicatorBar
import java.io.File
import java.io.FileOutputStream

/**
 * Save the words frequency per language into a [FrequencyDictionary] reading them from a dataset and save the
 * serialized model of the dictionary to file.
 *
 * The first argument is the filename of the input dataset.
 * The second argument is the output filename of the serialized model of the dictionary.
 */
fun main(args: Array<String>) {

  println("Reading dataset from '${args[0]}'...")
  val dataset: ArrayList<Example> = CorpusReader().read(File(args[0]))

  val dictionary = FrequencyDictionary()
  val fileSize: Int = File(args[0]).getNumOfLines()
  val progress = ProgressIndicatorBar(fileSize)

  println("Saving words occurrences...")

  dataset.forEach { (text, language) ->

    progress.tick()

    text.tokenize(maxTokensLength = 100).forEach { token ->
      dictionary.addOccurrence(word = token, language = language)
    }
  }

  dictionary.normalize()

  println("Saving dictionary to '${args[1]}'...")
  dictionary.dump(FileOutputStream(File(args[1])))
}
