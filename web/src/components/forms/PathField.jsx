import { Input } from "../ui/input";

export function PathField({ label, value, onChange, onBrowse, disabled }) {
  const hasValue =
    value !== undefined && value !== null && String(value) !== "";
  return (
    <div className="form-group">
      <div
        className={`file-input-wrapper md-field ${hasValue ? "has-value" : ""}`}
      >
        <Input
          className="md-input"
          value={value || ""}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          placeholder=" "
        />
        {label ? <label className="md-field-label">{label}</label> : null}
        <button
          type="button"
          className="file-picker-label"
          onClick={onBrowse}
          disabled={disabled}
          title="浏览本地路径"
        >
          <span className="material-icons">folder_open</span>
        </button>
      </div>
    </div>
  );
}
