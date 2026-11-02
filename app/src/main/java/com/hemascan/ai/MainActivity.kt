package com.hemascan.ai

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.rememberTransformableState
import androidx.compose.foundation.gestures.transformable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import java.io.File
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

val LABELS = listOf(
    Analyzer.UNCLASSIFIED, Analyzer.UNCERTAIN,
    "Érythrocyte normal", "Schizocyte", "Sphérocyte", "Elliptocyte", "Ovalocyte", "Microcyte", "Macrocyte",
    "Codocyte", "Dacryocyte", "Drépanocyte", "Acanthocyte", "Échinocyte", "Stomatocyte",
    "Neutrophile", "Lymphocyte", "Monocyte", "Éosinophile", "Basophile", "Plaquette", "Agrégat plaquettaire"
)

val Teal = Color(0xFF0F4C5C)
val Bg = Color(0xFFF3F6F8)
val Muted = Color(0xFF5F6F7A)
val Line = Color(0xFFE3E8EC)

data class Photo(val bmp: Bitmap, val cells: List<Cell> = emptyList())

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val ok = OpenCVLoader.initLocal()
        setContent {
            MaterialTheme(
                colorScheme = lightColorScheme(
                    primary = Teal, onPrimary = Color.White, secondary = Color(0xFF2E7D8C),
                    background = Bg, surface = Color.White, secondaryContainer = Color(0xFFD7E9ED)
                )
            ) { App(ok) }
        }
    }
}

fun loadBitmap(ctx: Context, uri: Uri): Bitmap? = try {
    val o = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    ctx.contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, o) }
    var ss = 1
    while (max(o.outWidth, o.outHeight) / ss > 1800) ss *= 2
    val o2 = BitmapFactory.Options().apply { inSampleSize = ss }
    ctx.contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, o2) }
} catch (e: Exception) { null }

fun exportCsv(ctx: Context, photos: List<Photo>) {
    val sb = StringBuilder("photo;classe;x;y;rayon_px\n")
    photos.forEachIndexed { i, p -> p.cells.forEach { sb.append("${i + 1};${it.label};${it.x.toInt()};${it.y.toInt()};${it.r.toInt()}\n") } }
    val f = File(ctx.cacheDir, "hemascan.csv"); f.writeText(sb.toString())
    val uri = FileProvider.getUriForFile(ctx, "${ctx.packageName}.fileprovider", f)
    val i = Intent(Intent.ACTION_SEND).apply {
        type = "text/csv"; putExtra(Intent.EXTRA_STREAM, uri); addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    ctx.startActivity(Intent.createChooser(i, "Exporter CSV"))
}

fun colorFor(label: String) = Color.hsv((label.hashCode() and 0x7fffffff) % 360f, 0.75f, 0.9f)

@Composable
fun Panel(modifier: Modifier = Modifier, content: @Composable ColumnScope.() -> Unit) {
    Card(
        modifier = modifier.fillMaxWidth(), shape = RoundedCornerShape(14.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp)
    ) { Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp), content = content) }
}

@Composable
fun Section(text: String) {
    Text(text.uppercase(), style = MaterialTheme.typography.labelMedium, color = Muted, fontWeight = FontWeight.SemiBold)
}

@Composable
fun Stat(label: String, value: String, color: Color, modifier: Modifier = Modifier) {
    Card(
        modifier = modifier, shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp)
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Text(value, style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold, color = color)
            Text(label, style = MaterialTheme.typography.labelMedium, color = Muted)
        }
    }
}

@Composable
fun App(cvOk: Boolean) {
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()
    val photos = remember { mutableStateListOf<Photo>() }
    var cur by remember { mutableStateOf(0) }
    var sel by remember { mutableStateOf<Int?>(null) }
    var filter by remember { mutableStateOf<String?>(null) }
    var show by remember { mutableStateOf(true) }
    var tab by remember { mutableStateOf(0) }
    var busy by remember { mutableStateOf(0) }
    var sens by remember { mutableStateOf(1f) }
    var camUri by remember { mutableStateOf<Uri?>(null) }

    fun run(idx: Int) {
        scope.launch {
            busy += 1
            val bmp = photos[idx].bmp
            val cells = withContext(Dispatchers.Default) { Analyzer.detect(bmp, sens) }
            if (idx < photos.size) photos[idx] = photos[idx].copy(cells = cells)
            sel = null; busy -= 1
        }
    }
    fun add(b: Bitmap) {
        photos.add(Photo(b)); cur = photos.lastIndex; sel = null; filter = null; run(photos.lastIndex)
    }

    val pick = rememberLauncherForActivityResult(ActivityResultContracts.GetMultipleContents()) { uris ->
        uris.forEach { u -> loadBitmap(ctx, u)?.let { add(it) } }
    }
    val cam = rememberLauncherForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        if (ok) camUri?.let { u -> loadBitmap(ctx, u)?.let { add(it) } }
    }

    Scaffold(
        containerColor = Bg,
        topBar = {
            Surface(color = Teal) {
                Column(modifier = Modifier.statusBarsPadding().fillMaxWidth().padding(horizontal = 16.dp, vertical = 12.dp)) {
                    Text("HemaScan AI", color = Color.White, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                    Text("Analyse de frottis · aide au comptage", color = Color(0xB3FFFFFF), style = MaterialTheme.typography.labelMedium)
                }
            }
        },
        bottomBar = {
            NavigationBar(containerColor = Color.White) {
                listOf("Analyse", "Résultats", "Comparaison").forEachIndexed { i, t ->
                    NavigationBarItem(
                        selected = tab == i, onClick = { tab = i },
                        icon = { Box(modifier = Modifier.size(10.dp).clip(CircleShape).background(if (tab == i) Teal else Color(0xFFB0BAC1))) },
                        label = { Text(t) }
                    )
                }
            }
        }
    ) { pad ->
        Column(
            modifier = Modifier.padding(pad).fillMaxSize().verticalScroll(rememberScrollState()).padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Surface(color = Color(0xFFFFF4E0), shape = RoundedCornerShape(10.dp)) {
                Text(
                    "Version non validée. Outil d'aide au comptage, ne pas utiliser pour un diagnostic. Classification manuelle.",
                    modifier = Modifier.padding(10.dp), style = MaterialTheme.typography.bodySmall, color = Color(0xFF7A4B00)
                )
            }
            if (!cvOk) Text("OpenCV n'a pas pu s'initialiser.", color = Color.Red)
            val p = photos.getOrNull(cur)
            when (tab) {
                0 -> {
                    Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        Button(onClick = {
                            val f = File(ctx.cacheDir, "cam_${System.currentTimeMillis()}.jpg")
                            val u = FileProvider.getUriForFile(ctx, "${ctx.packageName}.fileprovider", f)
                            camUri = u; cam.launch(u)
                        }, modifier = Modifier.weight(1f)) { Text("Prendre une photo") }
                        OutlinedButton(onClick = { pick.launch("image/*") }, modifier = Modifier.weight(1f)) { Text("Importer") }
                    }
                    if (p == null) {
                        Panel {
                            Text("Nouvelle analyse", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                            Text("1. Photographiez ou importez un champ microscopique.", style = MaterialTheme.typography.bodyMedium)
                            Text("2. La détection démarre automatiquement.", style = MaterialTheme.typography.bodyMedium)
                            Text("3. Vérifiez les cellules et classez-les au toucher.", style = MaterialTheme.typography.bodyMedium)
                            Text("4. Consultez les résultats et exportez en CSV.", style = MaterialTheme.typography.bodyMedium)
                            Text("Conseil : champ net et bien éclairé, cellules au centre.", style = MaterialTheme.typography.bodySmall, color = Muted)
                        }
                    } else {
                        if (photos.size > 1) LazyRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            items(photos.indices.toList()) { i ->
                                FilterChip(selected = i == cur, onClick = { cur = i; sel = null; filter = null }, label = { Text("Photo ${i + 1}") })
                            }
                        }
                        val unc = p.cells.count { it.label == Analyzer.UNCERTAIN }
                        val sug = p.cells.count { it.label == Analyzer.WBC || it.label == Analyzer.PLT }
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Stat("Cellules", if (busy > 0) "…" else "${p.cells.size}", Teal, Modifier.weight(1f))
                            Stat("Incertaines", if (busy > 0) "…" else "$unc", Color(0xFFB26A00), Modifier.weight(1f))
                            Stat("Suggestions", if (busy > 0) "…" else "$sug", Color(0xFF6A3FA0), Modifier.weight(1f))
                        }
                        if (busy > 0) LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                        Viewer(p, show, filter, sel) { sel = it }
                        Text("Pincer pour zoomer · double-tap pour réinitialiser · toucher une cellule pour la classer", style = MaterialTheme.typography.labelSmall, color = Muted)
                        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                            OutlinedButton(onClick = {
                                scope.launch {
                                    busy += 1
                                    val nb = withContext(Dispatchers.Default) { Analyzer.enhance(p.bmp) }
                                    photos[cur] = Photo(nb); busy -= 1; run(cur)
                                }
                            }, modifier = Modifier.weight(1f), enabled = busy == 0) { Text("Améliorer l'image") }
                            Button(onClick = { run(cur) }, modifier = Modifier.weight(1f), enabled = busy == 0) { Text("Relancer") }
                        }
                        Panel {
                            Section("Réglages d'affichage")
                            Text("Sensibilité de détection : %.1f".format(sens), style = MaterialTheme.typography.bodyMedium)
                            Slider(value = sens, onValueChange = { sens = it }, valueRange = 0.5f..1.5f, onValueChangeFinished = { run(cur) })
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Text("Afficher les annotations", modifier = Modifier.weight(1f))
                                Switch(checked = show, onCheckedChange = { show = it })
                            }
                            Section("Filtrer par catégorie")
                            LazyRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                                item { FilterChip(selected = filter == null, onClick = { filter = null }, label = { Text("Tout") }) }
                                items(p.cells.map { it.label }.distinct()) { l ->
                                    FilterChip(selected = filter == l, onClick = { filter = l }, label = { Text(l) })
                                }
                            }
                        }
                        val s = sel?.let { p.cells.getOrNull(it) }
                        if (s != null) Panel {
                            val conf = when {
                                s.conf >= 1f -> "Classée manuellement"
                                s.conf == 0f -> "Non classée"
                                else -> "Suggestion ~%.0f%% (heuristique, non validée)".format(s.conf * 100)
                            }
                            Section("Cellule sélectionnée")
                            Text("N° ${sel!! + 1} · ${s.label}", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                            Text(conf, style = MaterialTheme.typography.bodySmall, color = Muted)
                            Text("Choisir une classe :", style = MaterialTheme.typography.labelMedium)
                            LazyRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                                items(LABELS) { l ->
                                    AssistChip(onClick = {
                                        val nc = p.cells.toMutableList(); nc[sel!!] = s.copy(label = l, conf = 1f)
                                        photos[cur] = p.copy(cells = nc)
                                    }, label = { Text(l) })
                                }
                            }
                        }
                    }
                }
                1 -> {
                    if (p == null) Text("Aucune analyse pour l'instant.", color = Muted) else {
                        Section("Photo ${cur + 1}")
                        ResultTable(p.cells)
                        Button(onClick = { exportCsv(ctx, photos) }, modifier = Modifier.fillMaxWidth()) { Text("Exporter en CSV (toutes les photos)") }
                    }
                }
                else -> {
                    if (photos.isEmpty()) Text("Aucune photo.", color = Muted) else {
                        val maxN = max(1, photos.maxOf { it.cells.size })
                        Section("Par photo")
                        photos.forEachIndexed { i, ph ->
                            val unc = ph.cells.count { it.label == Analyzer.UNCERTAIN }
                            Panel {
                                Row {
                                    Text("Photo ${i + 1}", modifier = Modifier.weight(1f), fontWeight = FontWeight.SemiBold)
                                    Text("${ph.cells.size} cellules · $unc incertaines", style = MaterialTheme.typography.bodySmall, color = Muted)
                                }
                                Box(modifier = Modifier.fillMaxWidth().height(8.dp).clip(RoundedCornerShape(4.dp)).background(Line)) {
                                    Box(modifier = Modifier.fillMaxWidth(ph.cells.size / maxN.toFloat()).fillMaxHeight().background(Teal))
                                }
                            }
                        }
                        Section("Statistiques globales")
                        ResultTable(photos.flatMap { it.cells })
                        Text("Les cellules des champs qui se chevauchent ne sont pas dédupliquées.", style = MaterialTheme.typography.labelSmall, color = Muted)
                    }
                }
            }
        }
    }
}

@Composable
fun ResultTable(cells: List<Cell>) {
    val total = max(cells.size, 1)
    val groups = cells.groupBy { it.label }.toList().sortedByDescending { it.second.size }
    Panel {
        if (groups.isNotEmpty()) {
            Row(modifier = Modifier.fillMaxWidth().height(10.dp).clip(RoundedCornerShape(5.dp))) {
                groups.forEach { (l, v) -> Box(modifier = Modifier.weight(v.size.toFloat()).fillMaxHeight().background(colorFor(l))) }
            }
        }
        Row {
            Text("Classe", modifier = Modifier.weight(1f), style = MaterialTheme.typography.labelMedium, color = Muted)
            Text("N", modifier = Modifier.width(48.dp), style = MaterialTheme.typography.labelMedium, color = Muted)
            Text("%", modifier = Modifier.width(60.dp), style = MaterialTheme.typography.labelMedium, color = Muted)
        }
        Box(modifier = Modifier.fillMaxWidth().height(1.dp).background(Line))
        groups.forEach { (l, v) ->
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(modifier = Modifier.size(10.dp).clip(CircleShape).background(colorFor(l)))
                Spacer(Modifier.width(8.dp))
                Text(l, modifier = Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
                Text("${v.size}", modifier = Modifier.width(48.dp), style = MaterialTheme.typography.bodyMedium)
                Text("%.1f".format(100.0 * v.size / total), modifier = Modifier.width(60.dp), style = MaterialTheme.typography.bodyMedium)
            }
        }
        Box(modifier = Modifier.fillMaxWidth().height(1.dp).background(Line))
        Row {
            Text("Total", modifier = Modifier.weight(1f), fontWeight = FontWeight.Bold)
            Text("${cells.size}", modifier = Modifier.width(48.dp), fontWeight = FontWeight.Bold)
            Text("100", modifier = Modifier.width(60.dp), fontWeight = FontWeight.Bold)
        }
    }
}

@Composable
fun Viewer(p: Photo, show: Boolean, filter: String?, sel: Int?, onSel: (Int?) -> Unit) {
    var scale by remember { mutableStateOf(1f) }
    var off by remember { mutableStateOf(Offset.Zero) }
    var size by remember { mutableStateOf(IntSize.Zero) }
    val ts = rememberTransformableState { z, pan, _ -> scale = (scale * z).coerceIn(1f, 8f); off += pan }
    val img = remember(p.bmp) { p.bmp.asImageBitmap() }
    val bw = p.bmp.width.toFloat(); val bh = p.bmp.height.toFloat()
    fun fit(): Triple<Float, Float, Float> {
        val s = min(size.width / bw, size.height / bh)
        return Triple(s, (size.width - bw * s) / 2f, (size.height - bh * s) / 2f)
    }
    Box(
        modifier = Modifier.fillMaxWidth().height(380.dp).clip(RoundedCornerShape(14.dp)).clipToBounds().background(Color(0xFF10181C))
            .onSizeChanged { size = it }
            .transformable(ts)
            .graphicsLayer(scaleX = scale, scaleY = scale, translationX = off.x, translationY = off.y)
            .pointerInput(p, size) {
                detectTapGestures(
                    onDoubleTap = { scale = 1f; off = Offset.Zero },
                    onTap = { t ->
                        val (s, ox, oy) = fit()
                        var best: Int? = null; var bd = Float.MAX_VALUE
                        p.cells.forEachIndexed { i, c ->
                            val d = hypot(t.x - (ox + c.x * s), t.y - (oy + c.y * s))
                            if (d <= max(c.r * s, 24f) && d < bd) { bd = d; best = i }
                        }
                        onSel(best)
                    }
                )
            }
    ) {
        Image(bitmap = img, contentDescription = null, modifier = Modifier.fillMaxSize(), contentScale = ContentScale.Fit)
        Canvas(modifier = Modifier.fillMaxSize()) {
            if (!show || size == IntSize.Zero) return@Canvas
            val (s, ox, oy) = fit()
            val k = 1f / scale
            p.cells.forEachIndexed { i, c ->
                if (filter == null || c.label == filter) {
                    drawCircle(
                        color = if (i == sel) Color.White else colorFor(c.label), radius = c.r * s,
                        center = Offset(ox + c.x * s, oy + c.y * s), style = Stroke((if (i == sel) 5f else 2.5f) * k)
                    )
                }
            }
        }
    }
}
