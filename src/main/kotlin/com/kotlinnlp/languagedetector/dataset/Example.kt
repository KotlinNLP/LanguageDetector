/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.languagedetector.dataset

import com.kotlinnlp.linguisticdescription.language.Language

/**
 * An example to train or test a LanguageDetector.
 *
 * @property text the input text
 * @property language the language
 */
data class Example(val text: String, val language: Language)
