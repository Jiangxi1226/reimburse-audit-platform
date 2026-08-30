import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from reimbursement.invoice_parser import reconcile, _parse_json, to_audit_items, _infer_category

print('分项和=总额:', json.dumps(reconcile({'amount': 500, 'items': [{'name': '房费', 'amount': 300}, {'name': '早餐', 'amount': 200}]}), ensure_ascii=False))
print('分项和≠总额:', json.dumps(reconcile({'amount': 500, 'items': [{'name': '房费', 'amount': 320}]}), ensure_ascii=False))
print('无分项:', json.dumps(reconcile({'amount': 500, 'items': []}), ensure_ascii=False))
print('解析代码块:', _parse_json('前置\n```json\n{"amount":100}\n```后置'))
print('解析前后缀:', _parse_json('结果: {"amount": 99} 完'))
print('类别:', _infer_category('住宿发票'), _infer_category('航空行程单'), _infer_category('出租车票'))

# to_audit_items
parsed = {'extracted': {'invoice_type': '住宿发票', 'date': '2026-08-01', 'amount': 800, 'items': [{'name': '房费', 'amount': 500}, {'name': '早餐', 'amount': 300}]}}
print('audit_items:', json.dumps(to_audit_items(parsed), ensure_ascii=False))
