/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.languagedetector.utils

/**
 * A simple tokenizer which splits a text by spacing and pointing chars.
 */
class TextTokenizer {

  /**
   * The currently buffered token.
   */
  private val tokenBuffer = StringBuffer()

  /**
   * The list of tokens that is filled during the tokenization.
   */
  private val tokens = mutableListOf<String>()

  /**
   * Tokenize a text by spacing an punctuation chars.
   *
   * @param text the text to tokenize
   * @param maxTokensLength the max length of a token (longer tokens will be split)
   *
   * @return the list of tokens as list of [String]s
   */
  fun tokenize(text: String, maxTokensLength: Int): List<String> {
    require(maxTokensLength > 0)

    this.resetBuffers()

    text.forEachIndexed { i, char -> this.processChar(char = char, forceEnd = ((i + 1) % maxTokensLength) == 0) }

    if (this.tokenBuffer.isNotEmpty()) {
      this.addToken()
    }

    return tokens.toList()
  }

  /**
   * Process the given [char] of the input text.
   * If it is a spacing or punctuation char the current buffered token is added to the [tokens] list. Otherwise the
   * char is added to the buffered token.
   *
   * @param char a Char of the processing string
   * @param forceEnd a Boolean indicating if the current buffered token must end after processing the given [char]
   */
  private fun processChar(char: Char, forceEnd: Boolean) {

    if (!char.isLetter()) {

      this.addToken()

    } else {

      this.tokenBuffer.append(char)

      if (forceEnd) {
        this.addToken()
      }
    }
  }

  /**
   * Add the currently buffered token to the list of [tokens] and reset it.
   */
  private fun addToken() {

    val token: String = this.tokenBuffer.toString()

    if (token.isNotEmpty()) {
      this.tokens.add(token)
    }

    this.tokenBuffer.setLength(0)
  }

  /**
   * Reset all buffers (current token and tokens list).
   */
  private fun resetBuffers() {

    this.tokens.clear()
    this.tokenBuffer.setLength(0)
  }
}
