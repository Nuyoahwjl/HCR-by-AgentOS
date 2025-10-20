import pandas as pd
import os
import json

def get_health_check_info(card_number:int):
    # 加载 Excel 文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    file_path = os.path.join(project_root, 'data/dia.xlsx')
    df = pd.read_excel(file_path)

    # 假设“卡号”这一列名为 '卡号'，可根据实际情况修改
    filtered_df = df[df['卡号'] == card_number]

    if not filtered_df.empty:
        result_dict = filtered_df.iloc[0].to_dict()
        return json.dumps(result_dict, indent=4, ensure_ascii=False)
    else:
        return 0

if __name__ == "__main__":
    card_number = 18054423
    health_check_info = get_health_check_info(card_number)
    print(health_check_info)