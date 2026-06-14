import React from "react";
import { Button, Card, Form, Input, message, Modal, Select, Space, Table, Tag } from "antd";
import { UserPlus, RefreshCw, Trash2 } from "lucide-react";
import { api, apiError } from "../api/client";

export default function Students() {
  const [items, setItems] = React.useState<any[]>([]);
  const [open, setOpen] = React.useState(false);
  const [editing, setEditing] = React.useState<any | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [form] = Form.useForm();

  async function load() {
    setItems((await api.get("/users", { params: { role: "student" } })).data);
  }

  React.useEffect(() => {
    load().catch((error) => message.error(apiError(error)));
  }, []);

  function startCreate() {
    setEditing(null);
    form.resetFields();
    form.setFieldsValue({ role: "student" });
    setOpen(true);
  }

  function startEdit(record: any) {
    setEditing(record);
    form.setFieldsValue({ display_name: record.display_name, role: record.role, password: "" });
    setOpen(true);
  }

  async function submit(values: any) {
    setLoading(true);
    try {
      if (editing) {
        await api.put(`/users/${editing.username}`, values);
        message.success("学生信息已更新。");
      } else {
        await api.post("/users", values);
        message.success("学生账号已创建。");
      }
      setOpen(false);
      load();
    } catch (error) {
      message.error(apiError(error));
    } finally {
      setLoading(false);
    }
  }

  async function remove(username: string) {
    await api.delete(`/users/${username}`);
    message.success("学生账号已删除。");
    load();
  }

  return (
    <div className="pageStack">
      <section className="pageHero">
        <div>
          <span className="pageKicker">Student Management</span>
          <h1>学生信息管理</h1>
          <p>教师端维护学生演示账号、姓名和密码，并查看每个学生的预测使用情况。</p>
        </div>
        <div className="heroStats">
          <div><strong>{items.length}</strong><span>学生账号</span></div>
          <div><strong>{items.reduce((sum, item) => sum + Number(item.prediction_count || 0), 0)}</strong><span>预测记录</span></div>
          <div><strong>RBAC</strong><span>后端校验</span></div>
        </div>
      </section>

      <Card
        title="学生账号"
        extra={
          <Space>
            <Button icon={<RefreshCw size={16} />} onClick={load}>刷新</Button>
            <Button type="primary" icon={<UserPlus size={16} />} onClick={startCreate}>新增学生</Button>
          </Space>
        }
      >
        <Table
          className="nowrapTable"
          rowKey="username"
          dataSource={items}
          scroll={{ x: 920 }}
          columns={[
            { title: "账号", dataIndex: "username" },
            { title: "姓名", dataIndex: "display_name" },
            { title: "角色", dataIndex: "role", render: (v) => <Tag color="blue">{v}</Tag> },
            { title: "预测次数", dataIndex: "prediction_count" },
            { title: "最近预测", dataIndex: "last_prediction_at", render: (v) => v || "-" },
            { title: "创建时间", dataIndex: "created_at" },
            {
              title: "操作",
              render: (_, record) => (
                <Space>
                  <Button onClick={() => startEdit(record)}>编辑</Button>
                  <Button
                    danger
                    icon={<Trash2 size={16} />}
                    onClick={() => Modal.confirm({ title: "确认删除该学生账号？", content: "删除后该账号将无法登录。", onOk: () => remove(record.username) })}
                  >
                    删除
                  </Button>
                </Space>
              ),
            },
          ]}
        />
      </Card>

      <Modal title={editing ? "编辑学生" : "新增学生"} open={open} onCancel={() => setOpen(false)} footer={null}>
        <Form form={form} layout="vertical" onFinish={submit}>
          {!editing && (
            <Form.Item label="账号" name="username" rules={[{ required: true, message: "请输入账号" }]}>
              <Input placeholder="例如 student_01" />
            </Form.Item>
          )}
          <Form.Item label="姓名" name="display_name" rules={[{ required: true, message: "请输入姓名" }]}>
            <Input placeholder="学生姓名或备注" />
          </Form.Item>
          <Form.Item label={editing ? "重置密码" : "密码"} name="password" rules={editing ? [] : [{ required: true, message: "请输入密码" }]}>
            <Input.Password placeholder={editing ? "留空则不修改" : "默认可设为 123456"} />
          </Form.Item>
          <Form.Item label="角色" name="role" initialValue="student">
            <Select options={[{ value: "student", label: "学生" }, { value: "teacher", label: "教师" }]} />
          </Form.Item>
          <Button type="primary" htmlType="submit" loading={loading} block>{editing ? "保存修改" : "创建账号"}</Button>
        </Form>
      </Modal>
    </div>
  );
}
