pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        // The Polar BLE SDK is only published on JitPack.
        maven("https://jitpack.io") {
            content { includeGroup("com.github.polarofficial") }
        }
    }
}

rootProject.name = "open-holter-recorder"
include(":core", ":app")
