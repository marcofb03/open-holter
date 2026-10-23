package org.openholter.recorder.ui

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.ContextCompat
import org.openholter.recorder.Updater

val REQUIRED_PERMISSIONS = arrayOf(
    Manifest.permission.BLUETOOTH_SCAN,
    Manifest.permission.BLUETOOTH_CONNECT,
    Manifest.permission.POST_NOTIFICATIONS,
)

fun missingPermissions(context: Context): List<String> =
    REQUIRED_PERMISSIONS.filter { ContextCompat.checkSelfPermission(context, it) != PackageManager.PERMISSION_GRANTED }

private enum class Tab(val label: String, val glyph: String) {
    STATUS("Status", "♥"), DEVICE("Device", "⌁"), FILES("Files", "▤"), ABOUT("About", "ⓘ")
}

@Composable
fun HolterRoot() {
    val dark = isSystemInDarkTheme()
    val colors = if (dark) {
        darkColorScheme(primary = Color(0xFFEF9A9A), error = Color(0xFFFF8A80))
    } else {
        lightColorScheme(primary = Color(0xFFB71C1C), error = Color(0xFFC62828))
    }
    MaterialTheme(colorScheme = colors) {
        val context = LocalContext.current
        var tab by rememberSaveable { mutableIntStateOf(Tab.STATUS.ordinal) }
        var permissionsVersion by remember { mutableStateOf(0) }
        val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
            permissionsVersion++
        }
        val missing = remember(permissionsVersion) { missingPermissions(context) }
        // Only looks for a newer build; installing is always the user's choice (About → App updates).
        LaunchedEffect(Unit) { Updater.check() }
        Scaffold(
            bottomBar = {
                NavigationBar {
                    Tab.entries.forEach { t ->
                        NavigationBarItem(
                            selected = tab == t.ordinal,
                            onClick = { tab = t.ordinal },
                            icon = { Text(t.glyph, style = MaterialTheme.typography.titleLarge) },
                            label = { Text(t.label) },
                        )
                    }
                }
            },
        ) { padding ->
            Box(Modifier.padding(padding)) {
                when (Tab.entries[tab]) {
                    Tab.STATUS -> StatusScreen(
                        missingPermissions = missing,
                        onGrantPermissions = { launcher.launch(REQUIRED_PERMISSIONS) },
                        onChooseDevice = { tab = Tab.DEVICE.ordinal },
                    )
                    Tab.DEVICE -> DeviceScreen(
                        missingPermissions = missing,
                        onGrantPermissions = { launcher.launch(REQUIRED_PERMISSIONS) },
                        onStarted = { tab = Tab.STATUS.ordinal },
                    )
                    Tab.FILES -> FilesScreen()
                    Tab.ABOUT -> AboutScreen()
                }
            }
        }
    }
}
