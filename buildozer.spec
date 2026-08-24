[app]
source.dir = .
title = Ogham OCR
package.name = oghamapp
package.domain = org.ogham
source.include_exts = py,png,jpg,json,ttf,tflite

requirements = python3,kivy==2.3.1,numpy,pillow,plyer

android.permissions = READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE

android.api = 33
android.minapi = 24
android.ndk_api = 24
android.build_tools_version = 33.0.2
android.accept_sdk_license = True

[buildozer]
log_level = 2
warn_on_root = 1
