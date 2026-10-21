// Plain Kotlin/JVM module: everything that does not need Android (CSV format,
// clock mapping, reconnect state machine). Fast to unit-test on any JVM.
plugins {
    alias(libs.plugins.kotlin.jvm)
}

kotlin {
    jvmToolchain(21)
}

dependencies {
    testImplementation(libs.junit)
}
