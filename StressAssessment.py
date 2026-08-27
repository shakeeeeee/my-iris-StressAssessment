import streamlit as st
import plotly.express as px
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import joblib
import json
import os

# =========================================================================
# 1. 網頁基礎設定與快取加載
# =========================================================================
st.set_page_config(page_title="AI 智慧壓力風險評估系統", layout="centered")

@st.cache_resource
def load_assets():
    # 💡 獲取目前 pkl.py 所在的絕對資料夾路徑
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 💡 組合出絕對路徑，確保雲端伺服器 100% 找得到
    model_path = os.path.join(current_dir, 'lr_top8_model.pkl')
    json_path = os.path.join(current_dir, 'top8_features.json')
    
    # 開始讀取
    model = joblib.load(model_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        feature_names = json.load(f)
        
    return model, feature_names

try:
    model, top8_cols = load_assets()
except Exception as e:
    # 💡 這裡把具體的錯誤訊息 e 印出來，方便我們直接在線上看看到底是缺哪個檔
    st.error(f"🚨 找不到模型檔案或欄位清單！詳細錯誤原因：{e}")
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
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
   <script src="https://cdn.amcharts.com/lib/4/core.js"></script>
<script src="https://cdn.amcharts.com/lib/4/charts.js"></script>
<script src="https://cdn.amcharts.com/lib/4/themes/animated.js"></script>
    <style>
    body {{
      background-color: #0d1117; 
      margin: 0;
      padding: 0;
      overflow: hidden;
    }}
    
    #chartdiv {{
      width: 100%;
      height: 500px;
      background: #0d1117; 
      border-radius: 16px;
      border: 1px solid #30363d; 
      box-shadow: 0 0 30px rgba(0, 242, 254, 0.15); 
      position: relative;
    }}
    </style>
</head>
<body>
    
    <svg style="position: absolute; width: 0; height: 0;">
        <filter id="super-neon-glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="6" result="blur" />
            <feMerge>
                <feMergeNode in="blur"/>
                <feMergeNode in="SourceGraphic"/>
            </feMerge>
        </filter>
    </svg>

    <div id="chartdiv"></div>

    <script>
    am4core.ready(function() {{
    am4core.useTheme(am4themes_animated);

    var chartMin = 0;
    var chartMax = 100;
    var realModelScore = {final_gauge_score}; 

    var neonColors = {{
      healthy: "#00f2fe",   
      mild: "#00ff87",      
      moderate: "#fdae19",  
      severe: "#ff0055"     
    }};

    var data = {{
      score: realModelScore,
      gradingData: [
        {{ title: "HEALTHY", color: neonColors.healthy, lowScore: 0, highScore: 30 }},
        {{ title: "MILD", color: neonColors.mild, lowScore: 30, highScore: 55 }},
        {{ title: "MODERATE", color: neonColors.moderate, lowScore: 55, highScore: 80 }},
        {{ title: "SEVERE", color: neonColors.severe, lowScore: 80, highScore: 100 }}
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
    
    chart.padding(40, 40, 40, 40);
    chart.innerRadius = am4core.percent(75); 
    chart.resizable = true;

    var axis = chart.xAxes.push(new am4charts.ValueAxis());
    axis.min = chartMin;
    axis.max = chartMax;
    axis.strictMinMax = true;
    axis.renderer.radius = am4core.percent(100);
    axis.renderer.inside = true;
    axis.renderer.line.strokeOpacity = 0.1;
    axis.renderer.line.stroke = am4core.color("#fff");
    axis.renderer.ticks.template.disabled = false;
    axis.renderer.ticks.template.strokeOpacity = 0.4;
    axis.renderer.ticks.template.stroke = am4core.color("#58a6ff");
    axis.renderer.ticks.template.length = 5;
    axis.renderer.grid.template.disabled = true;
    
    axis.renderer.labels.template.radius = am4core.percent(12);
    axis.renderer.labels.template.fontSize = "0.85em";
    axis.renderer.labels.template.fill = am4core.color("#8b949e"); 

    var axis2 = chart.xAxes.push(new am4charts.ValueAxis());
    axis2.min = chartMin;
    axis2.max = chartMax;
    axis2.strictMinMax = true;
    axis2.renderer.labels.template.disabled = true;
    axis2.renderer.ticks.template.disabled = true;
    axis2.renderer.grid.template.disabled = false;
    axis2.renderer.grid.template.opacity = 0.05;

    
    for (let grading of data.gradingData) {{
      var range = axis2.axisRanges.create();
      var baseColor = am4core.color(grading.color);
      
      range.axisFill.fill = baseColor;
      range.axisFill.fillOpacity = 0.4; 
      range.axisFill.zIndex = -1;
      range.value = grading.lowScore;
      range.endValue = grading.highScore;
      range.grid.strokeOpacity = 0;
      range.stroke = baseColor;
      
      
    }}

    var matchingGrade = lookUpGrade(data.score, data.gradingData);

    
    var label = chart.radarContainer.createChild(am4core.Label);
    label.isMeasured = false;
    label.fontSize = "5.5em";
    label.fontFamily = "monospace"; 
    label.x = am4core.percent(50);
    label.paddingBottom = 25;
    label.horizontalCenter = "middle";
    label.verticalCenter = "bottom";
    label.text = data.score.toFixed(1);
    label.fill = am4core.color(matchingGrade.color);
    label.dom.setAttribute("filter", "url(#super-neon-glow)");

   
    var label2 = chart.radarContainer.createChild(am4core.Label);
    label2.isMeasured = false;
    label2.fontSize = "1.4em";
    label2.fontFamily = "monospace";
    label2.fontWeight = "bold";
    label2.horizontalCenter = "middle";
    label2.verticalCenter = "bottom";
    label2.text = "SYS_STATUS: " + matchingGrade.title;
    label2.fill = am4core.color(matchingGrade.color);
    label2.dom.setAttribute("filter", "url(#super-neon-glow)");

   
    var hand = chart.hands.push(new am4charts.ClockHand());
    hand.axis = axis2;
    hand.innerRadius = am4core.percent(50);
    hand.startWidth = 5;
    hand.endWidth = 1;
    hand.pin.disabled = true;
    hand.fill = am4core.color("#ffffff"); 
    hand.stroke = am4core.color("#ffffff");
    hand.dom.setAttribute("filter", "url(#super-neon-glow)");

    setTimeout(function() {{
        hand.showValue(data.score, 2000, am4core.ease.cubicOut);
    }}, 400);

    hand.events.on("positionchanged", function(){{
      var currentVal = axis2.positionToValue(hand.currentPosition);
      label.text = currentVal.toFixed(1);
      var matchingGrade = lookUpGrade(currentVal, data.gradingData);
      
      var targetColor = am4core.color(matchingGrade.color);
      label2.text = "SYS_STATUS: " + matchingGrade.title;
      label2.fill = targetColor;
      label.fill = targetColor;
    }});
    }});
    </script>
</body>
</html>
"""

# =========================================================================
# 5. 渲染網頁組件
# =========================================================================
st.subheader("⏱️ 即時壓力評估儀表：")
components.html(html_code, height=520)

real_stress_raw = (final_gauge_score / 100) * 9.0 + 1.0
st.info(f"💡 **AI 預測報告：** 當前受測者的即時壓力指標為 **{real_stress_raw:.2f} 分** (原始1~10分限制)。")


st.write("### 🧠 受測者基本資料與評估輸入")



