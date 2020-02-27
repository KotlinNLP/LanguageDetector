/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.languagedetector.dataset

import com.beust.klaxon.*
import com.kotlinnlp.linguisticdescription.language.Language
import java.io.File

/**
 * The helper to read a corpus from file.
 */
class CorpusReader {

  /**
   * All supported languages mapped to their iso-code (Unknown excluded).
   */
  private val supportedLanguages: Map<String, Language> =
    Language.values().filter { it != Language.Unknown }.associateBy { it.isoCode }

  /**
   * Read the given JSONL corpus [file] by lines.
   *
   * @param file the input file with a JSON example per line
   * @param maxLines the max number of lines to read, if null read the whole file (default = null)
   */
  fun read(file: File, maxLines: Int? = null): ArrayList<Example> {

    val examples = arrayListOf<Example>()
    val parser = Parser()
    var count = 0

    file.reader().forEachLine { line ->

      if (maxLines == null || count++ < maxLines) {

        val parsedExample = parser.parse(StringBuilder(line)) as JsonObject

        examples.add(Example(
          text = parsedExample.string("body")!!,
          language = this.getLanguage(parsedExample.string("language")!!)
        ))
      }
    }

    return examples
  }

  /**
   * Get the [Language] with the given [languageIsoCode].
   * [Language.Unknown] is returned if the given code doesn't matches any supported language.
   *
   * @param languageIsoCode the iso-code of a language (2 chars)
   *
   * @return the [Language] object with the given [languageIsoCode]
   */
  private fun getLanguage(languageIsoCode: String): Language {

    require(languageIsoCode.length == 2) { "Invalid language iso code (must be 2 chars long): $languageIsoCode" }

    return this.supportedLanguages[languageIsoCode] ?: Language.Unknown
  }
}
