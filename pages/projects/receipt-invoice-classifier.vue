<script setup lang="ts">
import * as tf from "@tensorflow/tfjs";

useHead({
  title: "Receipt-Invoice Classifier - dacxjo.github.io",
  meta: [
    {
      name: "description",
      content: "Receipt-Invoice Classifier demo using Tensorflow.js",
    },
    {
      name: "keywords",
      content: "ai, ml, machine learning, tensorflow, tensorflowjs, classifier",
    },
  ],
});

const image = ref<string | ArrayBuffer | null>(null);
const model = ref<tf.GraphModel<string | tf.io.IOHandler> | null>(null);
const classification = ref<"receipt" | "invoice" | null>(null);
onMounted(async () => {
  const modelUrl = "/receipt-or-invoice/model.json";
  model.value = await tf.loadGraphModel(modelUrl);
});

const handleFileSelect = (event: Event) => {
  const target = event.target as HTMLInputElement;
  const files = target.files;
  if (FileReader && files && files.length) {
    const fr = new FileReader();
    fr.onload = async () => {
      image.value = fr.result;
    };
    fr.onloadend = async () => {
      const img = new Image();
      img.width = 128;
      img.height = 128;
      img.src = image.value as string;
      const tensorImg = tf.browser
        //@ts-ignore
        .fromPixels(img)
        .toFloat()
        .expandDims();
      tf.tidy(() => {
        const rawModel = toRaw(model.value);
        const normalizedImage = tensorImg.div(255);
        const result = rawModel?.predict(normalizedImage);

        if (result) {
          //@ts-ignore
          const predictions = result.dataSync()[0];
          //@ts-ignore
          console.log(predictions);
          if (predictions > 0.5) {
            classification.value = "receipt";
          } else {
            classification.value = "invoice";
          }
        }
      });
    };
    fr.readAsDataURL(files[0]);
  }
};
</script>

<template>
  <div class="container mx-auto w-full h-screen p-5">
    <header class="p-8">
      <h1 class="text-4xl">Receipt-Invoice Classifier with Tensorflow</h1>
      <p class="mt-10">
        Binary classifier trained using Tensorflow and Keras using Transfer
        Learning from InceptionV3
      </p>
      <p class="mt-5">Try yourself, pick and invoice or receipt</p>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-5">
        <div
          class="flex flex-col-reverse justify-end xl:justify-normal xl:flex-row gap-5 md:col-span-2 col-span-1 mt-5 md:mt-0"
        >
          <div class="flex flex-col gap-5 mt-5">
            <input
              type="file"
              accept="image/png, image/jpeg"
              @change="handleFileSelect"
            />
            <img width="400" v-if="image" :src="image?.toString()" />
          </div>
          <h2 v-if="classification" class="text-2xl mt-5 lg:mt-0">
            Classification:
            <span class="text-green-400">{{ classification }}</span>
          </h2>
        </div>
        <div>
          <h2 class="text-2xl">Model info</h2>
          <ul class="list-disc p-5">
            <li>
              <b>Image augmentation params</b>: shear_range=0.3,
              brightness_range=[0.4, 1.5], vertical and horizontal flip
            </li>
            <li><b>RMSProp optimizer</b>: Initial learning rate 0.0001</li>
            <li><b>Epochs</b>: 20</li>
          </ul>
          <img src="/receipt-or-invoice/results/acc.png" alt="" />
          <img src="/receipt-or-invoice/results/loss.png" alt="" />
        </div>
      </div>
    </header>
  </div>
</template>
