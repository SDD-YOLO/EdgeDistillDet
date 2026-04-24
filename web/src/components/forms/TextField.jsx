import { Input } from "../ui/input";

export function TextField({ label, value, onChange, disabled }) {
  const hasValue = value !== undefined && value !== null && String(value) !== "";
  return (
    <div className="form-group">
      <div className={`md-field ${hasValue ? "has-value" : ""}`}>
        <Input className="md-input" value={value || ""} onChange={(e) => onChange(e.target.value)} disabled={disabled} placeholder=" " />
        <label className="md-field-label">{label}</label>
      </div>
    </div>
  );
}
