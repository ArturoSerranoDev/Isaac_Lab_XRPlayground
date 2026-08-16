using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace XRPlayground.Policies
{
    /// <summary>
    /// World-space MLP view: layer columns with discrete neuron dots (small layers)
    /// or compact activation strips (wide layers). Driven by <see cref="PolicyInferenceSnapshot"/>.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class PolicyNetworkVisualizer : MonoBehaviour
    {
        [Header("Layout")]
        public RectTransform contentRoot;
        [Tooltip("Layers wider than this use a heatmap strip instead of individual dots.")]
        public int maxDiscreteNeurons = 48;
        public float columnWidth = 72f;
        public float columnGap = 28f;
        public float plotHeight = 320f;
        public float neuronSize = 10f;

        [Header("Colors")]
        public Color idleColor = new Color(0.18f, 0.22f, 0.28f, 0.9f);
        public Color coldColor = new Color(0.12f, 0.35f, 0.55f, 1f);
        public Color hotColor = new Color(0.95f, 0.75f, 0.25f, 1f);
        public Color peakColor = new Color(1f, 0.45f, 0.2f, 1f);
        public Color labelColor = new Color(0.85f, 0.88f, 0.92f, 1f);

        [Header("Optional edges")]
        [Tooltip("Draw thin top-k edges between adjacent layers (skip when either layer is a strip).")]
        public bool drawTopKEdges = true;
        public int topKEdges = 6;
        public Color edgeColor = new Color(0.55f, 0.7f, 0.85f, 0.35f);

        struct LayerView
        {
            public RectTransform column;
            public Text label;
            public Image[] dots;
            public Image stripImage;
            public Texture2D stripTex;
            public int size;
            public bool useStrip;
            public float[] lastValues;
        }

        readonly List<LayerView> _layers = new();
        readonly List<Image> _edges = new();
        RectTransform _plotArea;
        RectTransform _edgeRoot;
        int[] _builtSizes;
        Font _font;

        public void Clear()
        {
            DestroyChildren(contentRoot);
            _layers.Clear();
            _edges.Clear();
            _plotArea = null;
            _edgeRoot = null;
            _builtSizes = null;
        }

        public void EnsureArchitecture(int[] layerSizes, string architectureLabel)
        {
            if (layerSizes == null || layerSizes.Length < 2)
                return;
            if (_builtSizes != null && SizesEqual(_builtSizes, layerSizes) && _layers.Count == layerSizes.Length)
                return;

            Build(layerSizes, architectureLabel);
        }

        public void ApplySnapshot(PolicyInferenceSnapshot snap)
        {
            if (snap == null || snap.LayerSizes == null || snap.LayerSizes.Length == 0)
                return;

            EnsureArchitecture(snap.LayerSizes, snap.ArchitectureLabel);
            if (_layers.Count == 0)
                return;

            for (int li = 0; li < _layers.Count; li++)
            {
                var view = _layers[li];
                float[] src = null;
                if (snap.LayerActivations != null && li < snap.LayerActivations.Length)
                    src = snap.LayerActivations[li];
                UpdateLayer(ref view, src);
                _layers[li] = view;
            }

            if (drawTopKEdges)
                RefreshEdges();
        }

        void Build(int[] layerSizes, string architectureLabel)
        {
            Clear();
            if (contentRoot == null)
                return;

            _font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            if (_font == null)
                _font = Resources.GetBuiltinResource<Font>("Arial.ttf");

            _builtSizes = (int[])layerSizes.Clone();

            float totalW = layerSizes.Length * columnWidth + (layerSizes.Length - 1) * columnGap;
            contentRoot.sizeDelta = new Vector2(Mathf.Max(totalW + 40f, contentRoot.sizeDelta.x), contentRoot.sizeDelta.y);

            _plotArea = CreateRect("Plot", contentRoot);
            SetAnchored(_plotArea, 20f, -70f, totalW, plotHeight + 40f);

            _edgeRoot = CreateRect("Edges", _plotArea);
            StretchFull(_edgeRoot);
            _edgeRoot.SetAsFirstSibling();

            float x = 0f;
            for (int i = 0; i < layerSizes.Length; i++)
            {
                int size = Mathf.Max(1, layerSizes[i]);
                bool useStrip = size > maxDiscreteNeurons;
                var col = CreateRect($"Layer_{i}", _plotArea);
                SetAnchored(col, x, 0f, columnWidth, plotHeight + 36f);

                var label = CreateText(col, "Label", LayerTitle(i, size, layerSizes.Length), 16, FontStyle.Bold);
                SetAnchored(label.rectTransform, 0f, 0f, columnWidth, 28f);
                label.alignment = TextAnchor.MiddleCenter;
                label.color = labelColor;

                var view = new LayerView
                {
                    column = col,
                    label = label,
                    size = size,
                    useStrip = useStrip,
                    lastValues = new float[size],
                };

                if (useStrip)
                {
                    var stripGo = CreateRect("Strip", col);
                    SetAnchored(stripGo, columnWidth * 0.28f, -34f, columnWidth * 0.44f, plotHeight);
                    var img = stripGo.gameObject.AddComponent<Image>();
                    img.color = Color.white;
                    view.stripImage = img;
                    view.stripTex = new Texture2D(1, size, TextureFormat.RGBA32, false)
                    {
                        wrapMode = TextureWrapMode.Clamp,
                        filterMode = FilterMode.Point,
                    };
                    img.sprite = Sprite.Create(
                        view.stripTex,
                        new Rect(0, 0, 1, size),
                        new Vector2(0.5f, 0.5f),
                        100f);
                    img.type = Image.Type.Simple;
                    // Fill idle
                    for (int n = 0; n < size; n++)
                        view.stripTex.SetPixel(0, n, idleColor);
                    view.stripTex.Apply(false);
                }
                else
                {
                    view.dots = new Image[size];
                    float usable = plotHeight - neuronSize;
                    for (int n = 0; n < size; n++)
                    {
                        float t = size == 1 ? 0.5f : n / (float)(size - 1);
                        float y = -34f - t * usable;
                        var dot = CreateRect($"N{n}", col);
                        SetAnchored(dot, (columnWidth - neuronSize) * 0.5f, y, neuronSize, neuronSize);
                        var img = dot.gameObject.AddComponent<Image>();
                        img.color = idleColor;
                        // Soft circle via radial fill looks rectangular without sprite; keep rounded via high fill.
                        view.dots[n] = img;
                    }
                }

                _layers.Add(view);
                x += columnWidth + columnGap;
            }
        }

        void UpdateLayer(ref LayerView view, float[] values)
        {
            float max = 1e-5f;
            if (values != null)
            {
                int n = Mathf.Min(view.size, values.Length);
                for (int i = 0; i < n; i++)
                {
                    float v = values[i];
                    view.lastValues[i] = v;
                    if (v > max) max = v;
                }
                for (int i = n; i < view.size; i++)
                    view.lastValues[i] = 0f;
            }
            else
            {
                for (int i = 0; i < view.size; i++)
                    view.lastValues[i] = 0f;
            }

            if (view.useStrip && view.stripTex != null)
            {
                for (int i = 0; i < view.size; i++)
                {
                    float intensity = Mathf.Clamp01(view.lastValues[i] / max);
                    view.stripTex.SetPixel(0, i, ColorFor(intensity));
                }
                view.stripTex.Apply(false);
            }
            else if (view.dots != null)
            {
                for (int i = 0; i < view.dots.Length; i++)
                {
                    float intensity = Mathf.Clamp01(view.lastValues[i] / max);
                    view.dots[i].color = ColorFor(intensity);
                }
            }
        }

        void RefreshEdges()
        {
            foreach (var e in _edges)
            {
                if (e != null)
                    Destroy(e.gameObject);
            }
            _edges.Clear();
            if (_edgeRoot == null || _layers.Count < 2 || topKEdges <= 0)
                return;

            for (int li = 0; li < _layers.Count - 1; li++)
            {
                var a = _layers[li];
                var b = _layers[li + 1];
                if (a.useStrip || b.useStrip)
                    continue; // skip dense strip edges for readability

                int k = Mathf.Min(topKEdges, a.size, b.size);
                var aIdx = TopIndices(a.lastValues, k);
                var bIdx = TopIndices(b.lastValues, k);
                int pairs = Mathf.Min(aIdx.Length, bIdx.Length);
                for (int p = 0; p < pairs; p++)
                {
                    Vector2 p0 = NeuronLocalCenter(a, aIdx[p]);
                    Vector2 p1 = NeuronLocalCenter(b, bIdx[p]);
                    // Convert to edge-root local
                    Vector2 w0 = a.column.anchoredPosition + p0;
                    Vector2 w1 = b.column.anchoredPosition + p1;
                    CreateEdge(w0, w1);
                }
            }
        }

        Vector2 NeuronLocalCenter(LayerView view, int neuronIndex)
        {
            if (view.dots != null && neuronIndex >= 0 && neuronIndex < view.dots.Length)
            {
                var rt = view.dots[neuronIndex].rectTransform;
                return rt.anchoredPosition + new Vector2(rt.sizeDelta.x * 0.5f, -rt.sizeDelta.y * 0.5f);
            }
            float usable = plotHeight - neuronSize;
            float t = view.size <= 1 ? 0.5f : neuronIndex / (float)(view.size - 1);
            float y = -34f - t * usable - neuronSize * 0.5f;
            return new Vector2(columnWidth * 0.5f, y);
        }

        void CreateEdge(Vector2 from, Vector2 to)
        {
            var go = CreateRect("Edge", _edgeRoot);
            var img = go.gameObject.AddComponent<Image>();
            img.color = edgeColor;
            Vector2 mid = (from + to) * 0.5f;
            Vector2 delta = to - from;
            float len = delta.magnitude;
            float angle = Mathf.Atan2(delta.y, delta.x) * Mathf.Rad2Deg;
            go.anchorMin = go.anchorMax = new Vector2(0f, 1f);
            go.pivot = new Vector2(0.5f, 0.5f);
            go.anchoredPosition = mid;
            go.sizeDelta = new Vector2(len, 1.5f);
            go.localRotation = Quaternion.Euler(0f, 0f, angle);
            _edges.Add(img);
        }

        Color ColorFor(float intensity)
        {
            if (intensity < 0.001f)
                return idleColor;
            if (intensity < 0.55f)
                return Color.Lerp(coldColor, hotColor, intensity / 0.55f);
            return Color.Lerp(hotColor, peakColor, (intensity - 0.55f) / 0.45f);
        }

        static int[] TopIndices(float[] values, int k)
        {
            if (values == null || values.Length == 0 || k <= 0)
                return System.Array.Empty<int>();
            k = Mathf.Min(k, values.Length);
            var idx = new int[values.Length];
            for (int i = 0; i < idx.Length; i++)
                idx[i] = i;
            System.Array.Sort(idx, (a, b) => values[b].CompareTo(values[a]));
            var top = new int[k];
            System.Array.Copy(idx, top, k);
            return top;
        }

        static string LayerTitle(int index, int size, int count)
        {
            if (index == 0) return $"in\n{size}";
            if (index == count - 1) return $"out\n{size}";
            return $"h{index}\n{size}";
        }

        static bool SizesEqual(int[] a, int[] b)
        {
            if (a.Length != b.Length) return false;
            for (int i = 0; i < a.Length; i++)
                if (a[i] != b[i]) return false;
            return true;
        }

        static RectTransform CreateRect(string name, Transform parent)
        {
            var go = new GameObject(name, typeof(RectTransform));
            go.transform.SetParent(parent, false);
            return go.GetComponent<RectTransform>();
        }

        Text CreateText(Transform parent, string name, string value, int size, FontStyle style)
        {
            var go = CreateRect(name, parent);
            var text = go.gameObject.AddComponent<Text>();
            text.text = value;
            text.fontSize = size;
            text.fontStyle = style;
            text.color = labelColor;
            text.font = _font;
            text.horizontalOverflow = HorizontalWrapMode.Wrap;
            text.verticalOverflow = VerticalWrapMode.Overflow;
            return text;
        }

        static void SetAnchored(RectTransform rt, float x, float y, float w, float h)
        {
            rt.anchorMin = new Vector2(0f, 1f);
            rt.anchorMax = new Vector2(0f, 1f);
            rt.pivot = new Vector2(0f, 1f);
            rt.anchoredPosition = new Vector2(x, y);
            rt.sizeDelta = new Vector2(w, h);
        }

        static void StretchFull(RectTransform rt)
        {
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = Vector2.one;
            rt.offsetMin = Vector2.zero;
            rt.offsetMax = Vector2.zero;
        }

        static void DestroyChildren(Transform root)
        {
            if (root == null) return;
            for (int i = root.childCount - 1; i >= 0; i--)
            {
                var c = root.GetChild(i).gameObject;
                if (Application.isPlaying)
                    Object.Destroy(c);
                else
                    Object.DestroyImmediate(c);
            }
        }

        void OnDestroy()
        {
            foreach (var layer in _layers)
            {
                if (layer.stripTex != null)
                    Destroy(layer.stripTex);
            }
        }
    }
}
