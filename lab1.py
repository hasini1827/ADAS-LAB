{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "451cd4ad-5a00-4a74-b364-6553e0cd9e9b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m4s\u001b[0m 4s/step\n",
      "[('n09193705', 'alp', np.float32(0.056015458)), ('n01910747', 'jellyfish', np.float32(0.028921988)), ('n09472597', 'volcano', np.float32(0.020509355))]\n",
      "Ambient Brightness: 123.22828707707629\n",
      "Headlights OFF\n"
     ]
    }
   ],
   "source": [
    "import cv2\n",
    "import numpy as np\n",
    "from tensorflow.keras.applications import MobileNetV2\n",
    "from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions\n",
    "#load pretrained cnn\n",
    "model = MobileNetV2(weights = 'imagenet')\n",
    "#read an image \n",
    "image = cv2.imread(\"day.jpg\")\n",
    "#resize the image \n",
    "img = cv2.resize(image,(224,224))\n",
    "#prepare the image \n",
    "x = np.expand_dims(img, axis=0)\n",
    "x= preprocess_input(x)\n",
    "#cnn prediction\n",
    "predictions = model.predict(x)\n",
    "\n",
    "print(decode_predictions(predictions, top=3)[0])\n",
    "\n",
    "#simulate brightness calculation\n",
    "gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n",
    "brightness = np.mean(gray)\n",
    "print(\"Ambient Brightness:\" ,brightness)\n",
    "if brightness<80:\n",
    "  print(\"Headlights ON\")\n",
    "else:\n",
    "  print(\"Headlights OFF\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4268b5fe-f5c3-4a2f-bff4-cc791e744b9a",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
