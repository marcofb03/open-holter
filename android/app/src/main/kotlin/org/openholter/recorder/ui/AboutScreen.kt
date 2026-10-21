package org.openholter.recorder.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.polar.sdk.api.PolarBleApiDefaultImpl
import org.openholter.recorder.BuildConfig

@Composable
fun AboutScreen() {
    val context = LocalContext.current
    var licence by remember { mutableStateOf<String?>(null) }

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("open-holter recorder ${BuildConfig.VERSION_NAME}", style = MaterialTheme.typography.headlineSmall)
        NotMedicalBanner()
        Text(
            "Records the single-lead ECG of a Polar H10 chest strap 24/7 into open CSV files on this phone, " +
                "with a symptom marker. Everything stays on the device until you export it.",
        )
        Text(
            "This app is an independent open-source project (MIT licence). It is not made, endorsed or " +
                "supported by Polar Electro Oy. \"Polar\" and \"H10\" are trademarks of Polar Electro Oy.",
            style = MaterialTheme.typography.bodySmall,
        )
        Text(
            "Uses the Polar BLE SDK ${runCatching { PolarBleApiDefaultImpl.versionInfo() }.getOrDefault("")}, " +
                "Copyright © Polar Electro Oy, under the Polar SDK licence below.",
            style = MaterialTheme.typography.bodySmall,
        )
        LICENCES.forEach { (title, asset) ->
            OutlinedButton(onClick = {
                licence = if (licence?.startsWith(title) == true) null
                else title + "\n\n" + context.assets.open(asset).bufferedReader().use { it.readText() }
            }) { Text(title) }
        }
        licence?.let { Text(it, fontSize = 11.sp) }
    }
}

private val LICENCES = listOf(
    "Polar BLE SDK licence" to "licenses/Polar_SDK_License.txt",
    "Polar SDK third-party software" to "licenses/Polar_SDK_ThirdPartySoftwareListing.txt",
    "open-holter licence (MIT)" to "licenses/open-holter_LICENSE.txt",
)
