---
title: Scene Rewind
emoji: 🐠
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: 'Time-travel with any image to any era '
---
# Scene Rewind

Scene Rewind is a fun and interactive web application that allows you to transform any image into a scene from a different era. Whether you want to see what your favorite photo would look like in the 80s, the 50s, or even the medieval times, Scene Rewind makes it easy and fun.

## Features

- **Era Transformation**: Upload any image and instantly see it transformed into a scene from a different era.
- **User-Friendly Interface**: Simple and intuitive interface that makes it easy for anyone to use.
- **Customizable Options**: Choose from a variety of eras and styles to get the perfect transformation.

## Tools Used

- **OpenAISDK**: Used to build the react agent architecture
- **LangFuse**: Used to follow the results of the scripts 
- **Replicate**: Used to create the images with "black-forest-labs/flux-kontext-pro"
- **SerpAPI**: To get better historical grounding
- **Gradio**: A powerful library for creating interactive web applications with minimal code.


## How to Use

1. **Upload an Image**: Click the upload button and select an image from your device.
2. **Choose an Era**: Select the era you want to transform your image into.
3. **Transform**: Click the transform button and see your image transformed into a scene from the chosen era.


## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for more details.
