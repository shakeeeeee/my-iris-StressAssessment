import streamlit as st
import plotly.express as px
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import joblib
import json

# =========================================================================
# 1. 網頁基礎設定與快取加載
# =========================================================================
st.set_page_config(page_title="AI 智慧壓力風險評估系統", layout="centered")

@st.cache_resource
def load_assets():
    model = joblib.load('lr_top8_model.pkl')
    with open('top8_features.json', 'r', encoding='utf-8') as f:
        feature_names = json.load(f)
    return model, feature_names

try:
    model, top8_cols = load_assets()
except Exception as e:
    st.error("🚨 找不到模型檔案或欄位清單！請確保先在 Jupyter 執行過存檔腳本。")
    st.stop()




st.title("🧠 AI 智慧健康大數據 ── 壓力風險評估儀表板")
st.write("本系統由你調校完美的精簡版線性迴歸模型驅動，即時透過 8 大核心生活行為指標推算受測者的壓力指數。")

# =========================================================================
# 2. 側邊欄：根據前 8 大特徵自動分流產生 UI
# =========================================================================
st.sidebar.header("📊 請輸入受測者健康與行為指標")
user_inputs = {}

name_mapping = {
    'bmi': '身體質量指數 (BMI)',
    'sleep_duration_hrs': '睡眠時數 (小時)',
    'sleep_quality_score': '睡眠品質分數 (0~100)',
    'alcohol_units_before_bed': '睡前飲酒量 (公升)',
    'cognitive_performance_score': '認知表現分數 (0~100)',
    'occupation_Retired': '是否退休 (Retired)',
    'occupation_Lawyer': '職業：律師 (Lawyer)',
    'occupation_Nurse': '職業：護理師 (Nurse)'
}

# 💡 核心線性轉換小工具：將原始值依比例精準投射到 0.2 ~ 0.8 區間
def rescale_to_model(val, min_val, max_val):
    if max_val == min_val:
        return 0.5
    return 0.2 + 0.6 * (val - min_val) / (max_val - min_val)

# =========================================================================
continuous_cols = []
categorical_cols = []

num_features = ['age', 'bmi', 'sleep_duration_hrs', 'sleep_quality_score', 'rem_percentage',
                'deep_sleep_percentage', 'sleep_latency_mins', 'wake_episodes_per_night',
                'caffeine_mg_before_bed', 'alcohol_units_before_bed', 'screen_time_before_bed_mins',
                'steps_that_day', 'nap_duration_mins', 'work_hours_that_day', 'heart_rate_resting_bpm',
                'room_temperature_celsius', 'weekend_sleep_diff_hrs', 'cognitive_performance_score']

# 分離欄位
for col in top8_cols:
    if col in num_features:
        continuous_cols.append(col)
    else:
        categorical_cols.append(col)

# 重新排序：連續型在前，類別型（是或否）在最下方
sorted_top8_cols = continuous_cols + categorical_cols

# =========================================================================
# 🎨 迴圈繪製側邊欄 (改用 sorted_top8_cols)
# =========================================================================
for col in sorted_top8_cols:
    display_name = name_mapping.get(col, col)
    
    # A. 連續型數值特徵
    if col in num_features:
        if col == 'cognitive_performance_score' or col == 'sleep_quality_score':
            raw_val = st.sidebar.slider(f"📈 {display_name}", min_value=0.0, max_value=100.0, value=75.0)
            user_inputs[col] = rescale_to_model(raw_val, 0.0, 100.0)
            
        elif col == 'sleep_duration_hrs':
            raw_val = st.sidebar.slider(f"📈 {display_name}", min_value=0.0, max_value=12.0, value=7.0, step=0.5)
            user_inputs[col] = rescale_to_model(raw_val, 0.0, 12.0)
            
        elif col == 'alcohol_units_before_bed':
            raw_val = st.sidebar.slider(f"📈 {display_name}", min_value=0.0, max_value=10.0, value=0.0, step=0.5)
            user_inputs[col] = rescale_to_model(raw_val, 0.0, 10.0)
            
        elif col == 'bmi':
            raw_val = st.sidebar.slider(f"📈 {display_name}", min_value=15.0, max_value=40.0, value=22.5, step=0.1)
            user_inputs[col] = rescale_to_model(raw_val, 15.0, 40.0)
            
        elif col == 'age':
            raw_val = st.sidebar.slider(f"📈 {display_name}", min_value=18.0, max_value=100.0, value=40.0, step=1.0)
            user_inputs[col] = rescale_to_model(raw_val, 18.0, 100.0)
            
        else:
            user_inputs[col] = st.sidebar.slider(f"📈 {display_name}", 0.2, 0.8, 0.5)
        
    # B. 二元型特徵 (是或否，排在下方)
    elif col in ['exercise_day', 'sleep_aid_used', 'shift_work', 'felt_rested']:
        choice = st.sidebar.selectbox(f"🔄 {display_name}", options=["否 (0)", "是 (1)"])
        user_inputs[col] = 1.0 if "是" in choice else 0.0
        
    # C. 其餘 One-Hot 分類特徵 (職業勾選等，排在最下方)
    else:
        choice = st.sidebar.checkbox(f"📍 {display_name}")
        user_inputs[col] = 0.8 if choice else 0.2
# =========================================================================
# 3. 後台即時 AI 預測與分數轉換邏輯
# =========================================================================
df_input = pd.DataFrame([user_inputs])[top8_cols]
y_pred_raw = model.predict(df_input)

# 將模型預測結果等比例放大投射到 amCharts 的 0~100 刻度
if y_pred_raw <= 1.0:
    final_gauge_score = ((y_pred_raw - 0.2) / 0.6) * 100
else:
    final_gauge_score = (y_pred_raw / 10.0) * 100

final_gauge_score = max(0.0, min(100.0, float(final_gauge_score.item())))
final_gauge_score = float(final_gauge_score)
# =========================================================================
# 4. 前端客製化：動態注入 amCharts 4 程式碼（對齊 0~100 刻度與四大健康燈號）
# =========================================================================
html_code = f"""
<style>
#chartdiv {{
  width: 100%;
  height: 500px;
  background-color: #ffffff;
  border-radius: 16px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}}
</style>

<!-- Resources -->
<script src="https://cdn.amcharts.com/lib/4/core.js"></script>
<script src="https://cdn.amcharts.com/lib/4/charts.js"></script>
<script src="https://cdn.amcharts.com/lib/4/themes/animated.js"></script>

<script>
am4core.ready(function() {{
am4core.useTheme(am4themes_animated);

var chartMin = 0;
var chartMax = 100;
var realModelScore = {final_gauge_score}; 

var data = {{
  score: realModelScore,
  gradingData: [
    {{ title: "HEALTHY", color: "#0f9747", lowScore: 0, highScore: 30 }},
    {{ title: "MILD", color: "#b0d136", lowScore: 30, highScore: 55 }},
    {{ title: "MODERATE", color: "#fdae19", lowScore: 55, highScore: 80 }},
    {{ title: "SEVERE", color: "#ee1f25", lowScore: 80, highScore: 100 }}
  ]
}};

function lookUpGrade(lookupScore, grades) {{
  for (var i = 0; i < grades.length; i++) {{
    if (grades[i].lowScore <= lookupScore && grades[i].highScore >= lookupScore) {{
      return grades[i];
    }}
  }}
  return grades[grades.length - 1];
}}

var chart = am4core.create("chartdiv", am4charts.GaugeChart);
chart.hiddenState.properties.opacity = 0;
chart.fontSize = 11;
chart.innerRadius = am4core.percent(80);
chart.resizable = true;

var axis = chart.xAxes.push(new am4charts.ValueAxis());
axis.min = chartMin;
axis.max = chartMax;
axis.strictMinMax = true;
axis.renderer.radius = am4core.percent(80);
axis.renderer.inside = true;
axis.renderer.line.strokeOpacity = 0.1;
axis.renderer.ticks.template.disabled = false;
axis.renderer.ticks.template.strokeOpacity = 1;
axis.renderer.ticks.template.strokeWidth = 0.5;
axis.renderer.ticks.template.length = 5;
axis.renderer.grid.template.disabled = true;
axis.renderer.labels.template.radius = am4core.percent(15);
axis.renderer.labels.template.fontSize = "0.9em";

var axis2 = chart.xAxes.push(new am4charts.ValueAxis());
axis2.min = chartMin;
axis2.max = chartMax;
axis2.strictMinMax = true;
axis2.renderer.labels.template.disabled = true;
axis2.renderer.ticks.template.disabled = true;
axis2.renderer.grid.template.disabled = false;
axis2.renderer.grid.template.opacity = 0.5;

for (let grading of data.gradingData) {{
  var range = axis2.axisRanges.create();
  range.axisFill.fill = am4core.color(grading.color);
  range.axisFill.fillOpacity = 0.8;
  range.axisFill.zIndex = -1;
  range.value = grading.lowScore;
  range.endValue = grading.highScore;
  range.grid.strokeOpacity = 0;
  range.stroke = am4core.color(grading.color).lighten(-0.1);
  range.label.inside = true;
  range.label.text = grading.title.toUpperCase();
  range.label.location = 0.5;
  range.label.radius = am4core.percent(10);
  range.label.fontSize = "0.9em";
}}

var matchingGrade = lookUpGrade(data.score, data.gradingData);

var label = chart.radarContainer.createChild(am4core.Label);
label.isMeasured = false;
label.fontSize = "5em";
label.x = am4core.percent(50);
label.paddingBottom = 15;
label.horizontalCenter = "middle";
label.verticalCenter = "bottom";
label.text = data.score.toFixed(1);
label.fill = am4core.color(matchingGrade.color);

var label2 = chart.radarContainer.createChild(am4core.Label);
label2.isMeasured = false;
label2.fontSize = "2em";
label2.horizontalCenter = "middle";
label2.verticalCenter = "bottom";
label2.text = matchingGrade.title.toUpperCase();
label2.fill = am4core.color(matchingGrade.color);

var hand = chart.hands.push(new am4charts.ClockHand());
hand.axis = axis2;
hand.innerRadius = am4core.percent(55);
hand.startWidth = 8;
hand.pin.disabled = true;
hand.fill = am4core.color("#444");
hand.stroke = am4core.color("#000");

setTimeout(function() {{
    hand.showValue(data.score, 1500, am4core.ease.cubicOut);
}}, 500);

hand.events.on("positionchanged", function(){{
  var currentVal = axis2.positionToValue(hand.currentPosition);
  label.text = currentVal.toFixed(1);
  var matchingGrade = lookUpGrade(currentVal, data.gradingData);
  label2.text = matchingGrade.title.toUpperCase();
  label2.fill = am4core.color(matchingGrade.color);
  label.fill = am4core.color(matchingGrade.color);
}});
}});
</script>
<div id="chartdiv"></div>
"""

# =========================================================================
# 5. 渲染網頁組件
# =========================================================================
st.subheader("⏱️ 即時壓力評估儀表：")
components.html(html_code, height=520)

real_stress_raw = (final_gauge_score / 100) * 9.0 + 1.0
st.info(f"💡 **AI 預測報告：** 當前受測者的即時壓力指標為 **{real_stress_raw:.2f} 分** (原始1~10分限制)。")


st.write("### 🧠 受測者基本資料與評估輸入")



