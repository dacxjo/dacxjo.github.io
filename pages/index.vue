<script lang="ts" setup>
useMeta({
    title:'David Blandon | Dev | Student | Common human',
    meta:[
        {
          description: 'This is my personal website... but is a WIP'
        }
    ]
})
</script>

<template>
    <main class="h-screen bg-teal-700 flex flex-col items-center justify-center">
        <h1 class="text-white text-sm ">David Blandon</h1>
        <a href="https://twitter.com/dacxj0" class="underline text-white " target="_blank">@dacxj0</a>
        <div draggable="true" class="flex w-1/3 flex-col items-center justify-center mt-5">
            <div class="w-full h-10 bg-gray-300 rounded-t-lg p-2">
                <span class="text-sm text-gray-700">Creating personal website...</span>
            </div>
            <div class="w-full h-64 bg-teal-900 rounded-b-lg p-2">
                <p class="text-white text-sm">
                    $dacxj0: Generating pages
                    <span></span>
                </p>
            </div>
        </div>
    </main>
</template>

<style scoped>
[draggable="true"] {
    cursor: move;
}
p > span::before {
    animation: dots 2s linear infinite;
    content: "";
}

@keyframes dots {
    0% , 20% {
        content: ".";
    }
    40% {
        content: "..";
    }

    60% {
        content: "...";
    }

    90%, 100% {
        content: "";
    }
}
</style>