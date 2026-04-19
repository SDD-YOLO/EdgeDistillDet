import { useEffect, useState } from "react";
import { getStoredTeamId, getStoredToken, setAuthStorage } from "../api/client.js";
import { createTeam, fetchVersion, listTeams, login, register } from "../api/saasApi.js";

export default function SaasBar({ toast }) {
  const [saas, setSaas] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [teamName, setTeamName] = useState("");
  const [teams, setTeams] = useState([]);
  const [teamId, setTeamId] = useState(getStoredTeamId());

  useEffect(() => {
    fetchVersion()
      .then((v) => setSaas(!!v.saas))
      .catch(() => setSaas(false));
  }, []);

  useEffect(() => {
    if (!saas || !getStoredToken()) return;
    listTeams()
      .then((r) => setTeams(r.teams || []))
      .catch(() => {});
  }, [saas, teamId]);

  if (!saas) return null;

  const onLogin = async () => {
    try {
      await login(email, password);
      const r = await listTeams();
      setTeams(r.teams || []);
      toast("已登录", "success");
    } catch (e) {
      toast(e.message, "error");
    }
  };

  const onRegister = async () => {
    try {
      await register(email, password);
      toast("注册成功，已登录", "success");
    } catch (e) {
      toast(e.message, "error");
    }
  };

  const onCreateTeam = async () => {
    try {
      const r = await createTeam(teamName);
      const id = r.team?.id;
      if (id) {
        setAuthStorage({ teamId: id });
        setTeamId(id);
      }
      const lt = await listTeams();
      setTeams(lt.teams || []);
      toast("团队已创建", "success");
    } catch (e) {
      toast(e.message, "error");
    }
  };

  const onTeamChange = (e) => {
    const id = e.target.value;
    setTeamId(id);
    setAuthStorage({ teamId: id || null });
  };

  const logout = () => {
    setAuthStorage({ accessToken: "", teamId: "" });
    setTeams([]);
    setTeamId("");
    toast("已退出", "info");
  };

  return (
    <div className="saas-bar">
      {!getStoredToken() ? (
        <>
          <input
            type="email"
            placeholder="邮箱"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="saas-input"
          />
          <input
            type="password"
            placeholder="密码"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="saas-input"
          />
          <button type="button" className="saas-btn" onClick={onLogin}>
            登录
          </button>
          <button type="button" className="saas-btn secondary" onClick={onRegister}>
            注册
          </button>
        </>
      ) : (
        <>
          <select value={teamId} onChange={onTeamChange} className="saas-select" aria-label="当前团队">
            <option value="">选择团队…</option>
            {teams.map((t) => (
              <option key={t.id} value={t.id}>
                {t.name}
              </option>
            ))}
          </select>
          <input
            type="text"
            placeholder="新团队名称"
            value={teamName}
            onChange={(e) => setTeamName(e.target.value)}
            className="saas-input"
          />
          <button type="button" className="saas-btn secondary" onClick={onCreateTeam}>
            创建团队
          </button>
          <button type="button" className="saas-btn ghost" onClick={logout}>
            退出
          </button>
        </>
      )}
    </div>
  );
}
