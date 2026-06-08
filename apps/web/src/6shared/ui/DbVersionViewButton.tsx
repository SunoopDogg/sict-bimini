'use client';

import { Eye } from 'lucide-react';

import { useLocale } from '@/6shared/i18n';
import { Button } from '@/6shared/ui/primitive/button';

interface DbVersionViewButtonProps {
  version: string;
  onView: (version: string) => void;
}

/**
 * Trailing eye-icon action for a DB version row: opens the version-scoped
 * contents viewer. Shared by VersionSelect (1height) and CreateVersionPanel
 * (2height) so the affordance stays identical across both planes. Lives in
 * 6shared so neither feature slice has to import the other.
 */
export function DbVersionViewButton({
  version,
  onView,
}: DbVersionViewButtonProps) {
  const { t } = useLocale();
  return (
    <Button
      type="button"
      variant="ghost"
      size="icon"
      className="text-muted-foreground hover:text-foreground h-7 w-7"
      onClick={() => onView(version)}
      aria-label={t.bimAttr.view}
      title={t.bimAttr.view}
    >
      <Eye />
    </Button>
  );
}
